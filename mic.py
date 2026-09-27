"""
Round-trip audio through the codec and listen to the result.

    uv run python mic.py                      # record 4 s from the mic
    uv run python mic.py --codebooks 2        # same, at a quarter the bitrate
    uv run python mic.py --input voice.wav    # use a file instead

Recording and playback go through ffmpeg and afplay, so there is nothing
to install. macOS asks for microphone access the first time.
"""

import argparse
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from checkpoint import load_mimi_decoder, load_mimi_encoder

HF_REPO = "kyutai/moshiko-pytorch-bf16"
MIMI_CHECKPOINT = "tokenizer-e351c8d8-checkpoint125.safetensors"
SAMPLE_RATE = 24000
OUT_DIR = Path("out")


def list_input_devices():
    # ffmpeg prints its device list to stderr and then exits non-zero
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation",
         "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    )
    print(proc.stderr.strip())


def record(seconds, device, path):
    print(f"recording {seconds}s from device {device} — speak now")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "avfoundation", "-i", f":{device}",
         "-t", str(seconds), "-ar", str(SAMPLE_RATE), "-ac", "1",
         "-sample_fmt", "s16", str(path)],
        check=True,
    )


def to_mono_24k(source, path):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-sample_fmt", "s16", str(path)],
        check=True,
    )


def read_wav(path):
    with wave.open(str(path)) as f:
        if f.getsampwidth() != 2 or f.getnchannels() != 1:
            raise ValueError("expected 16-bit mono")
        raw = f.readframes(f.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def write_wav(path, samples):
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--codebooks", type=int, default=8,
                        help="1 to 32; Moshi uses 8")
    parser.add_argument("--input", help="audio file to use instead of the mic")
    parser.add_argument("--device", default="0", help="avfoundation audio index")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--no-play", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    OUT_DIR.mkdir(exist_ok=True)
    original = OUT_DIR / "original.wav"
    rebuilt = OUT_DIR / "reconstructed.wav"

    if args.input:
        to_mono_24k(args.input, original)
    else:
        record(args.seconds, args.device, original)

    samples = read_wav(original)
    if not len(samples):
        sys.exit("no audio captured — try --list-devices to find the right index")

    print("loading Mimi weights")
    path = hf_hub_download(HF_REPO, MIMI_CHECKPOINT)
    encoder = load_mimi_encoder(path)
    decoder = load_mimi_decoder(path)

    waveform = torch.from_numpy(samples).view(1, 1, -1)

    with torch.no_grad():
        codes = encoder(waveform, num_codebooks=args.codebooks)
        reconstruction = decoder(codes)[0, 0, : len(samples)]

    write_wav(rebuilt, reconstruction.numpy())

    seconds = len(samples) / SAMPLE_RATE
    bits = codes.numel() * 11
    print(f"\n{seconds:.2f}s  ->  {tuple(codes.shape[1:])} codes"
          f"  ->  {bits / seconds / 1000:.2f} kbps")
    print(f"compression: {len(samples) * 16 / bits:.0f}x versus 16-bit PCM")
    print(f"\n  {original}\n  {rebuilt}")

    if not args.no_play:
        for label, file in (("original", original), ("reconstructed", rebuilt)):
            print(f"playing {label}")
            subprocess.run(["afplay", str(file)], check=False)


if __name__ == "__main__":
    main()
