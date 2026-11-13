"""Train the diffusion model on CIFAR-10 (32x32 colour) and save the weights.

    uv run python train_diffusion.py                # 20 epochs
    uv run python train_diffusion.py --epochs 3     # quicker, blurrier images

Writes app/diffusion_cifar10.pt (used by /generate_diffusion) and
samples/diffusion_samples.png (a grid of generated images).
"""
import argparse
import copy
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

from app.diffusion_model import Diffusion, UNet
from app.image_generators import (
    DIFFUSION_STEPS,
    DIFFUSION_WEIGHTS,
    SHAPE,
    TRAIN_TRANSFORM,
    to_png_bytes,
)

SAMPLES_PATH = Path("samples/diffusion_samples.png")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(loader, epochs, device, weights_path=DIFFUSION_WEIGHTS, samples_path=SAMPLES_PATH,
          lr=2e-4, ema_decay=0.995):
    model = UNet().to(device)
    ema = copy.deepcopy(model).eval()           # smoothed copy of the weights, samples look better
    diffusion = Diffusion(T=DIFFUSION_STEPS, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"UNet parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, _ in loader:                      # labels are not used
            x = x.to(device)
            optimizer.zero_grad()
            loss = diffusion.loss(model, x)      # predict the noise that was added
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():                # exponential moving average of the weights
                for e, p in zip(ema.parameters(), model.parameters()):
                    e.mul_(ema_decay).add_(p.detach(), alpha=1 - ema_decay)
            running += loss.item()
        print(f"epoch {epoch + 1}/{epochs}  loss={running / len(loader):.4f}")

    weights_path = Path(weights_path)
    torch.save({k: v.cpu() for k, v in ema.state_dict().items()}, weights_path)
    print(f"saved weights to {weights_path}")

    samples = diffusion.sample(ema, (16, *SHAPE))
    if samples_path is not None:
        samples_path = Path(samples_path)
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        samples_path.write_bytes(to_png_bytes(samples))
        print(f"saved sample grid to {samples_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device}")
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=TRAIN_TRANSFORM)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train(loader, args.epochs, device)


if __name__ == "__main__":
    main()
