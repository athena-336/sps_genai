"""Train the energy-based model on MNIST (32x32) with contrastive divergence.

    uv run python train_ebm.py                # 10 epochs
    uv run python train_ebm.py --epochs 3     # quicker, rougher digits

EBMs train slowly (every step runs a Langevin chain), so use more epochs if the
samples in samples/ebm_samples.png still look like noise.

Each training step:
  1. take "fake" images from a replay buffer and lower their energy with
     Langevin dynamics (gradient descent on the *image*, weights frozen),
  2. update the *weights* so real images get low energy and fake ones high energy.

Writes app/ebm_mnist.pt (used by /generate_ebm) and samples/ebm_samples.png.
"""
import argparse
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

from app.ebm_model import EnergyCNN, ReplayBuffer, langevin_sample
from app.image_generators import EBM_WEIGHTS, SHAPE, TRANSFORM, to_png_bytes

SAMPLES_PATH = Path("samples/ebm_samples.png")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(loader, epochs, device, langevin_steps=60, weights_path=EBM_WEIGHTS,
          samples_path=SAMPLES_PATH, lr=1e-4, alpha=0.1, real_noise=0.005):
    model = EnergyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.0, 0.999))
    buffer = ReplayBuffer(shape=SHAPE)
    print(f"EnergyCNN parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(epochs):
        totals = {"loss": 0.0, "real": 0.0, "fake": 0.0}
        for x, _ in loader:
            x = x.to(device)
            real = (x + real_noise * torch.randn_like(x)).clamp(-1, 1)

            # (1) Sample fake images: gradients flow to the IMAGES, not to the weights.
            fake = langevin_sample(model, buffer.sample(real.shape[0]).to(device),
                                   steps=langevin_steps)
            buffer.push(fake)

            # (2) Update the WEIGHTS: lower energy of real images, raise it for fake ones.
            model.train()
            real_energy, fake_energy = model(torch.cat([real, fake])).chunk(2)
            cdiv = real_energy.mean() - fake_energy.mean()
            reg = alpha * (real_energy ** 2 + fake_energy ** 2).mean()  # keeps energies bounded
            loss = cdiv + reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            totals["loss"] += loss.item()
            totals["real"] += real_energy.mean().item()
            totals["fake"] += fake_energy.mean().item()
        n = len(loader)
        print(f"epoch {epoch + 1}/{epochs}  loss={totals['loss'] / n:.4f}  "
              f"E(real)={totals['real'] / n:.3f}  E(fake)={totals['fake'] / n:.3f}")

    weights_path = Path(weights_path)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, weights_path)
    print(f"saved weights to {weights_path}")

    start = torch.rand(16, *SHAPE, device=device) * 2 - 1
    samples = langevin_sample(model, start, steps=256)
    if samples_path is not None:
        samples_path = Path(samples_path)
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        samples_path.write_bytes(to_png_bytes(samples))
        print(f"saved sample grid to {samples_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--langevin-steps", type=int, default=60)
    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device}")
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=TRANSFORM)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train(loader, args.epochs, device, langevin_steps=args.langevin_steps)


if __name__ == "__main__":
    main()
