import torch

from seanet import SimpleSEANetEncoder


def main():
    model = SimpleSEANetEncoder()
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    # 1 second of mono audio at 24kHz: (batch, channels, samples)
    waveform = torch.randn(2, 1, 24000)
    with torch.no_grad():
        latents = model(waveform)

    # total stride 4*5*6*8 = 960 -> 24000 / 960 = 25 frames/sec
    print(f"input:  {tuple(waveform.shape)}")
    print(f"output: {tuple(latents.shape)}")


if __name__ == "__main__":
    main()
