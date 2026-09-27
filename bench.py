"""
Repeatable benchmark for the codec, with a saved baseline to compare against.

    uv run python bench.py --save "eager fp32"     # record a baseline
    ... make a change ...
    uv run python bench.py --label "kv cache"      # compare against it

Every run also checks that the output has not changed. A speedup that
alters the codes is not a speedup, so correctness is reported first and
a mismatch is called out loudly.
"""

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from checkpoint import load_mimi_decoder, load_mimi_encoder

HF_REPO = "kyutai/moshiko-pytorch-bf16"
MIMI_CHECKPOINT = "tokenizer-e351c8d8-checkpoint125.safetensors"
SAMPLE_RATE = 24000
CHUNK = 1920
BUDGET_MS = CHUNK / SAMPLE_RATE * 1000
RESULTS = Path("bench.json")

STAGES = [
    "encode seanet", "encode transformer", "encode downsample", "quantize",
    "dequantize", "decode upsample", "decode transformer", "decode seanet",
]


def synchronizer(device):
    if device == "cuda":
        return torch.cuda.synchronize
    if device == "mps":
        return torch.mps.synchronize

    return lambda: None


def fixed_input(device, chunks):
    """Deterministic, and shaped like speech rather than noise."""
    generator = torch.Generator().manual_seed(0)
    length = CHUNK * chunks
    t = torch.arange(length) / SAMPLE_RATE
    signal = torch.zeros(length)

    for harmonic in range(1, 40):
        frequency = 120.0 * harmonic
        if frequency > 11000:
            break
        glide = t / t[-1]
        formants = (
            torch.exp(-((frequency - (730 + (270 - 730) * glide)) / 180) ** 2)
            + 0.7 * torch.exp(-((frequency - (1090 + (2290 - 1090) * glide)) / 240) ** 2)
        )
        signal += formants * torch.sin(2 * torch.pi * frequency * t) / harmonic ** 0.5

    signal += 0.002 * torch.randn(length, generator=generator)

    return (0.5 * signal / signal.abs().max()).view(1, 1, -1).to(device)


def run_once(encoder, decoder, x, codebooks, sync, marks):
    def timed(name, fn, *args, **kwargs):
        sync()
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        sync()
        marks[name] = marks.get(name, []) + [(time.perf_counter() - start) * 1000]
        return result

    a = timed("encode seanet", encoder.seanet, x)
    b = timed("encode transformer", encoder.transformer, a)
    c = timed("encode downsample", encoder.downsample, b)
    codes = timed("quantize", lambda z: encoder.quantizer.encode(
        z, num_codebooks=codebooks), c)

    latents = timed("dequantize", decoder.quantizer.decode, codes)
    latents = timed("decode upsample", decoder.upsample, latents)
    latents = timed("decode transformer", decoder.transformer, latents)
    audio = timed("decode seanet", decoder.seanet, latents)

    return codes, audio


def fingerprint(codes, audio):
    """Cheap, stable signature of what the model produced."""
    return {
        "codeSum": int(codes.sum()),
        "codeShape": list(codes.shape),
        "audioMean": round(float(audio.float().mean()), 8),
        "audioAbsSum": round(float(audio.float().abs().sum()), 4),
    }


def summarise(marks):
    out = {}
    for stage, values in marks.items():
        values = sorted(values)
        out[stage] = {
            "min": round(values[0], 3),
            "p50": round(statistics.median(values), 3),
            "p95": round(values[min(len(values) - 1, int(len(values) * 0.95))], 3),
        }
    totals = sorted(sum(v) for v in zip(*(marks[s] for s in marks)))
    out["TOTAL"] = {
        "min": round(totals[0], 3),
        "p50": round(statistics.median(totals), 3),
        "p95": round(totals[min(len(totals) - 1, int(len(totals) * 0.95))], 3),
    }
    return out


def delta(now, before):
    if before in (None, 0):
        return ""
    change = (now - before) / before * 100
    arrow = "faster" if change < 0 else "slower"
    if abs(change) < 1.5:
        return f"{before:8.2f}      same"
    return f"{before:8.2f}  {abs(change):5.1f}% {arrow}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--codebooks", type=int, default=8)
    parser.add_argument("--chunks", type=int, default=1,
                        help="80 ms chunks per step; 1 is the streaming case")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--label", default="unlabelled")
    parser.add_argument("--save", metavar="LABEL",
                        help="store this run as the baseline")
    parser.add_argument("--history", action="store_true",
                        help="print past runs and exit")
    args = parser.parse_args()

    store = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}

    if args.history:
        for entry in store.get("history", []):
            print(f"{entry['when'][:19]}  {entry['total']:7.2f} ms  {entry['label']}")
        return

    torch.set_num_threads(args.threads)
    path = hf_hub_download(HF_REPO, MIMI_CHECKPOINT)
    encoder = load_mimi_encoder(path).to(args.device)
    decoder = load_mimi_decoder(path).to(args.device)

    x = fixed_input(args.device, args.chunks)
    sync = synchronizer(args.device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            decoder(encoder(x, num_codebooks=args.codebooks))

        marks = {}
        for _ in range(args.iters):
            codes, audio = run_once(encoder, decoder, x, args.codebooks, sync, marks)

    stats = summarise(marks)
    signature = fingerprint(codes, audio)
    label = args.save or args.label
    baseline = store.get("baseline")

    # ---- correctness first -------------------------------------------------
    print()
    if baseline is None:
        print("  output    no baseline yet")
    elif baseline["fingerprint"] == signature:
        print("  output    unchanged from baseline")
    else:
        print("  output    CHANGED FROM BASELINE")
        for key, value in signature.items():
            was = baseline["fingerprint"].get(key)
            if was != value:
                print(f"              {key}: {was} -> {value}")

    # ---- timings -----------------------------------------------------------
    budget = BUDGET_MS * args.chunks
    print()
    head = f"  {'stage':22s} {'min':>8s} {'p50':>8s} {'p95':>8s}"
    if baseline:
        head += f"   {'was(min)':>8s}"
    print(head)
    print("  " + "-" * (len(head) - 2))

    for stage in STAGES + ["TOTAL"]:
        s = stats[stage]
        line = f"  {stage:22s} {s['min']:8.2f} {s['p50']:8.2f} {s['p95']:8.2f}"
        if baseline:
            line += f"   {delta(s['min'], baseline['stats'].get(stage, {}).get('min'))}"
        print(line)

    total = stats["TOTAL"]["min"]
    print()
    print(f"  budget {budget:.0f} ms   using {total / budget * 100:.0f}%"
          f"   headroom {budget - total:.1f} ms"
          f"   {budget / total:.2f}x realtime")
    print(f"  {args.device} · {args.threads} thread(s) · {args.codebooks} codebooks"
          f" · {args.chunks} chunk(s) · {platform.machine()}")

    entry = {
        "when": datetime.now(timezone.utc).isoformat(),
        "label": label, "total": total, "stats": stats,
        "fingerprint": signature,
        "config": {"device": args.device, "threads": args.threads,
                   "codebooks": args.codebooks, "chunks": args.chunks},
    }
    store.setdefault("history", []).append(entry)
    if args.save or baseline is None:
        store["baseline"] = entry
        print(f"\n  saved as baseline: {label}")
    RESULTS.write_text(json.dumps(store, indent=2))


if __name__ == "__main__":
    main()
