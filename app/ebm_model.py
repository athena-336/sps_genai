"""Energy-based model for images (Assignment 4, Part 1).

Convention used throughout: the network outputs an *energy* E(x).  Low energy
means "looks like a real image", so  p(x) is proportional to exp(-E(x)).

Training (contrastive divergence)
    loss = E(real).mean() - E(fake).mean()  +  alpha * (E(real)^2 + E(fake)^2).mean()
    -> pushes the energy of real images down and the energy of generated ones up.

Sampling (Langevin dynamics)
    x <- x - step_size * grad_x E(x) + noise
    -> here the gradient is taken with respect to the INPUT IMAGE, not the
       weights.  The weights are frozen and only x changes.
"""
import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class EnergyCNN(nn.Module):
    """1x32x32 image -> one scalar energy per image."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 5, stride=2, padding=4), Swish(),  # 18x18
            nn.Conv2d(16, 32, 3, stride=2, padding=1), Swish(),           # 9x9
            nn.Conv2d(32, 64, 3, stride=2, padding=1), Swish(),           # 5x5
            nn.Conv2d(64, 64, 3, stride=2, padding=1), Swish(),           # 3x3
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 64), Swish(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def langevin_sample(
    model: nn.Module,
    x: torch.Tensor,
    steps: int = 60,
    step_size: float = 10.0,
    noise_std: float = 0.005,
    grad_clip: float = 0.03,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Run Langevin dynamics starting from ``x`` and return the lowered-energy images.

    Which tensors get gradients?  Only the image ``x``.  We freeze every model
    parameter (``requires_grad_(False)``) and use ``torch.autograd.grad`` on
    ``x`` so nothing is accumulated in the weights' ``.grad`` fields.
    """
    was_training = model.training
    model.eval()
    param_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    x = x.detach().clone()
    try:
        with torch.enable_grad():  # also works if the caller is inside torch.no_grad()
            for _ in range(steps):
                noise = torch.randn(x.shape, generator=generator, device=x.device)
                x = (x + noise_std * noise).clamp(-1.0, 1.0)
                x.requires_grad_(True)                       # <- gradients w.r.t. the image
                energy = model(x)
                (grad,) = torch.autograd.grad(energy.sum(), x)
                grad = grad.clamp(-grad_clip, grad_clip)     # keep each step small and stable
                x = (x - step_size * grad).clamp(-1.0, 1.0).detach()  # descend the energy
    finally:
        for p, flag in zip(model.parameters(), param_flags):
            p.requires_grad_(flag)
        model.train(was_training)
    return x


class ReplayBuffer:
    """Keeps past samples so Langevin chains do not have to restart from noise each step."""

    def __init__(self, shape=(1, 32, 32), max_size: int = 8192, fresh_prob: float = 0.05,
                 init_size: int = 128):
        self.shape = tuple(shape)
        self.max_size = max_size
        self.fresh_prob = fresh_prob
        self.buffer = self._noise(init_size)

    def _noise(self, n: int) -> torch.Tensor:
        return torch.rand((n, *self.shape)) * 2 - 1

    def sample(self, batch_size: int) -> torch.Tensor:
        """~95% of the batch comes from the buffer, ~5% restarts from fresh noise."""
        n_fresh = int(torch.distributions.Binomial(batch_size, torch.tensor(self.fresh_prob)).sample())
        idx = torch.randint(0, len(self.buffer), (batch_size - n_fresh,))
        return torch.cat([self._noise(n_fresh), self.buffer[idx]], dim=0)

    def push(self, samples: torch.Tensor) -> None:
        self.buffer = torch.cat([samples.detach().cpu(), self.buffer], dim=0)[: self.max_size]
