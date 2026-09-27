"""
Live latency dashboard for the codec.

    uv run python profiler.py                  # synthetic audio, best device found
    uv run python profiler.py --device cpu --threads 1
    uv run python profiler.py --input voice.wav
    uv run python profiler.py --mic             # live from the microphone

Then open http://localhost:8765.

The loop feeds one 80 ms chunk at a time, which is the shape a streaming
encoder would see. Timings are therefore representative even though this
model carries no streaming state, so the audio it produces would click at
every chunk boundary. Latency, not output, is the point here.
"""

import argparse
import json
import math
import subprocess
import threading
import time
from collections import deque
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from checkpoint import load_mimi_decoder, load_mimi_encoder

HF_REPO = "kyutai/moshiko-pytorch-bf16"
MIMI_CHECKPOINT = "tokenizer-e351c8d8-checkpoint125.safetensors"
SAMPLE_RATE = 24000
CHUNK = 1920                      # one frame at 12.5 Hz
BUDGET_MS = CHUNK / SAMPLE_RATE * 1000
HISTORY = 400

STAGES = [
    "encode seanet", "encode transformer", "encode downsample", "quantize",
    "dequantize", "decode upsample", "decode transformer", "decode seanet",
]

state = {
    "status": "starting", "device": "cpu", "codebooks": 8,
    "source": "synthetic", "budgetMs": BUDGET_MS, "stages": STAGES,
    "steps": 0, "latest": {}, "totals": [], "codes": [],
}
config = {"codebooks": 8, "running": True}
lock = threading.Lock()
history = deque(maxlen=HISTORY)


def synchronizer(device):
    """
    Accelerator work is queued asynchronously, so a stage has to be waited
    on before its wall time means anything. CPU work is already synchronous.
    """
    if device == "cuda":
        return torch.cuda.synchronize
    if device == "mps":
        return torch.mps.synchronize

    return lambda: None


class Stopwatch:
    """Times each stage, synchronising so device work is actually attributed."""

    def __init__(self, device):
        self.sync = synchronizer(device)
        self.marks = {}

    def run(self, name, fn, *args, **kwargs):
        self.sync()
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        self.sync()
        self.marks[name] = (time.perf_counter() - start) * 1000
        return result


def synthetic_audio(seconds=4):
    t = torch.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    glide = t / t[-1]
    out = torch.zeros_like(t)
    for h in range(1, 40):
        f = 120.0 * h
        if f > 11000:
            break
        amp = (torch.exp(-((f - (730 + (270 - 730) * glide)) / 180) ** 2)
               + 0.7 * torch.exp(-((f - (1090 + (2290 - 1090) * glide)) / 240) ** 2))
        out += amp * torch.sin(2 * math.pi * f * t) / h ** 0.5
    return 0.5 * out / out.abs().max()


def read_wav_via_ffmpeg(source):
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source),
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout
    return torch.from_numpy(np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0)


def mic_chunks():
    """Raw 16-bit mono frames straight off the microphone."""
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "avfoundation", "-i", ":0",
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "-"],
        stdout=subprocess.PIPE,
    )
    try:
        while True:
            raw = proc.stdout.read(CHUNK * 2)
            if len(raw) < CHUNK * 2:
                return
            yield torch.from_numpy(np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0)
    finally:
        proc.kill()


def looping_chunks(samples):
    position = 0
    while True:
        if position + CHUNK > len(samples):
            position = 0
        yield samples[position:position + CHUNK]
        position += CHUNK


def worker(args):
    with lock:
        state["status"] = "loading weights"
    path = hf_hub_download(HF_REPO, MIMI_CHECKPOINT)

    device = args.device
    encoder = load_mimi_encoder(path).to(device)
    decoder = load_mimi_decoder(path).to(device)

    if args.mic:
        source, chunks = "microphone", mic_chunks()
    elif args.input:
        source, chunks = Path(args.input).name, looping_chunks(read_wav_via_ffmpeg(args.input))
    else:
        source, chunks = "synthetic", looping_chunks(synthetic_audio())

    with lock:
        state.update(status="running", device=device, source=source)

    warm = torch.zeros(1, 1, CHUNK, device=device)
    with torch.inference_mode():
        for _ in range(3):
            decoder(encoder(warm, num_codebooks=8))

    for chunk in chunks:
        if not config["running"]:
            time.sleep(0.05)
            continue

        codebooks = config["codebooks"]
        x = chunk.view(1, 1, -1).to(device)
        watch = Stopwatch(device)

        with torch.inference_mode():
            a = watch.run("encode seanet", encoder.seanet, x)
            b = watch.run("encode transformer", encoder.transformer, a)
            c = watch.run("encode downsample", encoder.downsample, b)
            codes = watch.run("quantize", partial(
                encoder.quantizer.encode, num_codebooks=codebooks), c)

            latents = watch.run("dequantize", decoder.quantizer.decode, codes)
            latents = watch.run("decode upsample", decoder.upsample, latents)
            latents = watch.run("decode transformer", decoder.transformer, latents)
            watch.run("decode seanet", decoder.seanet, latents)

        total = sum(watch.marks.values())
        history.append(watch.marks)

        with lock:
            state["steps"] += 1
            state["latest"] = watch.marks
            state["codebooks"] = codebooks
            state["codes"] = codes[0, :, 0].tolist()
            state["totals"] = [round(sum(m.values()), 3) for m in history]
            state["percentiles"] = percentiles()
            state["totalMs"] = round(total, 3)

        # Pace to realtime when we are faster than the budget, so the
        # numbers reflect a stream rather than a benchmark.
        slack = BUDGET_MS / 1000 - total / 1000
        if slack > 0 and source != "microphone":
            time.sleep(slack)


def percentiles():
    if not history:
        return {}
    out = {}
    for stage in STAGES:
        values = sorted(m.get(stage, 0.0) for m in history)
        out[stage] = {
            "p50": round(values[len(values) // 2], 3),
            "p95": round(values[min(len(values) - 1, int(len(values) * 0.95))], 3),
            "max": round(values[-1], 3),
        }
    totals = sorted(sum(m.values()) for m in history)
    out["_total"] = {
        "p50": round(totals[len(totals) // 2], 3),
        "p95": round(totals[min(len(totals) - 1, int(len(totals) * 0.95))], 3),
        "max": round(totals[-1], 3),
    }
    return out


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            body = Path("dashboard.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    with lock:
                        payload = json.dumps(state)
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.12)
            except (BrokenPipeError, ConnectionResetError):
                return

        self.send_error(404)

    def do_POST(self):
        if self.path != "/config":
            return self.send_error(404)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or "{}")
        if "codebooks" in body:
            config["codebooks"] = max(1, min(32, int(body["codebooks"])))
        if "running" in body:
            config["running"] = bool(body["running"])
        self.send_response(204)
        self.end_headers()


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=default_device(),
                        choices=["cpu", "mps", "cuda"])
    parser.add_argument("--threads", type=int, default=0,
                        help="torch CPU threads; fewer is often faster here")
    parser.add_argument("--input", help="audio file to loop")
    parser.add_argument("--mic", action="store_true", help="live microphone input")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    threading.Thread(target=worker, args=(args,), daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"dashboard: http://localhost:{args.port}  (ctrl-c to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
