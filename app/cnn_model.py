"""CNN for CIFAR10 image classification (Assignment 2, Part 1)."""
import torch.nn as nn

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class CNN(nn.Module):
    """Input: RGB image 3x64x64 -> logits for 10 classes.

    Shapes:
      3x64x64 -> conv(16) 16x64x64 -> pool 16x32x32
              -> conv(32) 32x32x32 -> pool 32x16x16
              -> flatten 8192 -> fc 100 -> fc 10
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
