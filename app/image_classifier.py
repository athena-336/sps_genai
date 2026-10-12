"""Loads the trained CNN and classifies images (Assignment 2, Part 2)."""
import io
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

from app.cnn_model import CLASSES, CNN

WEIGHTS_PATH = Path(__file__).parent / "cnn_cifar10.pt"

# Must match the preprocessing used in train_cnn.py.
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)


class ImageClassifier:
    def __init__(self, weights_path: Path = WEIGHTS_PATH):
        self.model = CNN()
        self.available = weights_path.exists()
        if self.available:
            self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()

    def predict(self, image_bytes: bytes, top_k: int = 3) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = TRANSFORM(image).unsqueeze(0)  # [1, 3, 64, 64]
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
        top = torch.topk(probs, k=min(top_k, len(CLASSES)))
        top_predictions = [
            {"class": CLASSES[int(i)], "confidence": round(float(p), 4)}
            for p, i in zip(top.values, top.indices)
        ]
        return {
            "class": top_predictions[0]["class"],
            "confidence": top_predictions[0]["confidence"],
            "top_predictions": top_predictions,
        }
