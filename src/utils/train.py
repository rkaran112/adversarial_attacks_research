"""Minimal training and clean-evaluation loops."""

import torch
import torch.nn as nn

from ..metrics import accuracy


def train_model(
    model: nn.Module,
    loader,
    epochs: int = 1,
    lr: float = 1e-3,
    device: str = "cpu",
    max_batches: int = None,
) -> nn.Module:
    """Train ``model`` on ``loader`` with Adam + cross-entropy.

    ``max_batches`` caps the number of batches per epoch (useful for smoke
    tests); ``None`` trains on the full loader.
    """
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for i, (images, labels) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def evaluate_clean(model: nn.Module, loader, device: str = "cpu", max_batches: int = None) -> float:
    """Return top-1 accuracy of ``model`` on clean data from ``loader``."""
    model = model.to(device)
    model.eval()
    correct = 0.0
    total = 0
    for i, (images, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        correct += (logits.argmax(1) == labels).float().sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0
