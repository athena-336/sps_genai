"""Train the CNN on CIFAR10 (resized to 64x64) and save weights to app/cnn_cifar10.pt.

Usage:  uv run python train_cnn.py
"""
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from app.cnn_model import CNN
from app.image_classifier import TRANSFORM, WEIGHTS_PATH

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=TRANSFORM
    )
    test_ds = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=TRANSFORM
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)  # forward + loss
            loss.backward()  # backward
            optimizer.step()  # update weights and biases
            running_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(dim=1) == y).sum().item()
        print(
            f"epoch {epoch + 1}/{EPOCHS}  "
            f"train_loss={running_loss / len(train_loader):.4f}  "
            f"test_acc={correct / len(test_ds):.4f}"
        )

    model.cpu()
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Saved weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
