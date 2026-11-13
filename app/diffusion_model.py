"""A small DDPM-style diffusion model (Assignment 4, Part 1).

Two pieces:

* ``UNet``      - the noise-prediction network eps_theta(x_t, t).
* ``Diffusion`` - the forward (noising) process, the training loss and the
                  reverse (denoising) sampling loop.

Images are 1x32x32 and scaled to [-1, 1].
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Noise-prediction network
# --------------------------------------------------------------------------- #
class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of the integer timestep, followed by a small MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return self.mlp(torch.cat([args.sin(), args.cos()], dim=1))


class ResBlock(nn.Module):
    """GroupNorm -> SiLU -> Conv, with the timestep embedding added in the middle."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """Predicts the noise that was added to x_t.  32x32 -> 16x16 -> 8x8 -> back up."""

    def __init__(self, in_channels: int = 1, base: int = 32, t_dim: int = 128):
        super().__init__()
        self.time_embed = TimeEmbedding(t_dim)
        self.inc = nn.Conv2d(in_channels, base, 3, padding=1)

        self.down1 = ResBlock(base, base, t_dim)                        # 32x32
        self.pool1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)  # -> 16x16
        self.down2 = ResBlock(base * 2, base * 2, t_dim)                # 16x16
        self.pool2 = nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1)  # -> 8x8

        self.mid1 = ResBlock(base * 2, base * 2, t_dim)                 # 8x8
        self.mid2 = ResBlock(base * 2, base * 2, t_dim)

        self.up2 = nn.Conv2d(base * 2, base * 2, 3, padding=1)          # after x2 upsample
        self.dec2 = ResBlock(base * 4, base * 2, t_dim)                 # concat skip from down2
        self.up1 = nn.Conv2d(base * 2, base, 3, padding=1)
        self.dec1 = ResBlock(base * 2, base, t_dim)                     # concat skip from down1

        self.out_norm = nn.GroupNorm(8, base)
        self.out = nn.Conv2d(base, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.inc(x)
        s1 = self.down1(h, t_emb)                       # 32x32
        s2 = self.down2(self.pool1(s1), t_emb)          # 16x16
        h = self.mid2(self.mid1(self.pool2(s2), t_emb), t_emb)  # 8x8

        h = self.up2(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = self.dec2(torch.cat([h, s2], dim=1), t_emb)
        h = self.up1(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = self.dec1(torch.cat([h, s1], dim=1), t_emb)
        return self.out(F.silu(self.out_norm(h)))


# --------------------------------------------------------------------------- #
# Diffusion process
# --------------------------------------------------------------------------- #
def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal, 2021)."""
    x = torch.linspace(0, T, T + 1)
    alpha_bar = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-4, 0.999)


class Diffusion:
    """Forward noising, training loss, and ancestral (DDPM) sampling."""

    def __init__(self, T: int = 200, device="cpu"):
        self.T = T
        self.device = torch.device(device)

        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])

        # q(x_t | x_0) = N( sqrt(alpha_bar_t) x_0 , (1 - alpha_bar_t) I )
        self.sqrt_alpha_bar = alpha_bar.sqrt().to(self.device)
        self.sqrt_one_minus_alpha_bar = (1 - alpha_bar).sqrt().to(self.device)

        # posterior q(x_{t-1} | x_t, x_0): mean = coef1 * x_0 + coef2 * x_t
        self.post_coef1 = (betas * alpha_bar_prev.sqrt() / (1 - alpha_bar)).to(self.device)
        self.post_coef2 = ((1 - alpha_bar_prev) * alphas.sqrt() / (1 - alpha_bar)).to(self.device)
        self.post_std = (betas * (1 - alpha_bar_prev) / (1 - alpha_bar)).sqrt().to(self.device)

    @staticmethod
    def _gather(values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return values[t].reshape(-1, 1, 1, 1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward process: jump straight from x_0 to x_t."""
        return (self._gather(self.sqrt_alpha_bar, t) * x0
                + self._gather(self.sqrt_one_minus_alpha_bar, t) * noise)

    def loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        """Simple DDPM loss: MSE between the true noise and the predicted noise."""
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        return F.mse_loss(model(x_t, t), noise)

    @torch.no_grad()
    def sample(self, model: nn.Module, shape, generator: torch.Generator = None) -> torch.Tensor:
        """Start from pure noise and denoise for T steps. Returns images in [-1, 1]."""
        model.eval()
        x = torch.randn(shape, generator=generator, device=self.device)
        for step in reversed(range(self.T)):
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            eps = model(x, t)
            # Estimate x_0 from the predicted noise, clip to the valid pixel range,
            # then take the posterior mean of x_{t-1}.
            x0 = (x - self._gather(self.sqrt_one_minus_alpha_bar, t) * eps) \
                / self._gather(self.sqrt_alpha_bar, t)
            x0 = x0.clamp(-1.0, 1.0)
            mean = self._gather(self.post_coef1, t) * x0 + self._gather(self.post_coef2, t) * x
            if step > 0:
                noise = torch.randn(shape, generator=generator, device=self.device)
                x = mean + self._gather(self.post_std, t) * noise
            else:
                x = mean
        return x
