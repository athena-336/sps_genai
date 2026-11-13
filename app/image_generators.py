"""Diffusion and energy-based image generators used by the API (Assignment 4, Part 1)."""
import io
import threading
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import make_grid

from app.diffusion_model import Diffusion, UNet
from app.ebm_model import EnergyCNN, langevin_sample

IMAGE_SIZE = 32
SHAPE = (1, IMAGE_SIZE, IMAGE_SIZE)
DIFFUSION_STEPS = 200

DIFFUSION_WEIGHTS = Path(__file__).parent / "diffusion_mnist.pt"
EBM_WEIGHTS = Path(__file__).parent / "ebm_mnist.pt"

# Training-time preprocessing (shared by train_diffusion.py and train_ebm.py):
# MNIST digits resized to 32x32 and scaled from [0, 1] to [-1, 1].
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def to_png_bytes(images: torch.Tensor, scale: int = 4) -> bytes:
    """Tile a batch of [-1, 1] images into one grid and return it as PNG bytes."""
    n = images.shape[0]
    grid = make_grid(images.cpu(), nrow=min(n, 4), padding=2, normalize=True, value_range=(-1, 1))
    array = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    img = Image.fromarray(array)
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _generator(seed: Optional[int]) -> Optional[torch.Generator]:
    return None if seed is None else torch.Generator().manual_seed(seed)


class DiffusionGenerator:
    def __init__(self, weights_path: Path = DIFFUSION_WEIGHTS):
        weights_path = Path(weights_path)
        self.model = UNet()
        self.diffusion = Diffusion(T=DIFFUSION_STEPS, device="cpu")
        self.available = weights_path.exists()
        if self.available:
            self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()

    def generate(self, num_images: int = 4, seed: Optional[int] = None) -> bytes:
        images = self.diffusion.sample(self.model, (num_images, *SHAPE), generator=_generator(seed))
        return to_png_bytes(images)


class EBMGenerator:
    def __init__(self, weights_path: Path = EBM_WEIGHTS):
        weights_path = Path(weights_path)
        self.model = EnergyCNN()
        self.available = weights_path.exists()
        if self.available:
            self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()
        # langevin_sample temporarily freezes the weights, so serialise requests.
        self._lock = threading.Lock()

    def generate(self, num_images: int = 4, steps: int = 256, seed: Optional[int] = None) -> bytes:
        g = _generator(seed)
        start = torch.rand((num_images, *SHAPE), generator=g) * 2 - 1  # uniform noise
        with self._lock:
            images = langevin_sample(self.model, start, steps=steps, generator=g)
        return to_png_bytes(images)
