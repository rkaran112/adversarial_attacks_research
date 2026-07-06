"""Shared fixtures: a tiny trainable model and synthetic image batches."""

import os
import sys

import pytest
import torch
import torch.nn as nn

# Make the project root importable so `import src...` works under pytest.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TinyNet(nn.Module):
    """A minimal conv classifier for 1x8x8 inputs, 3 classes."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.fc = nn.Linear(4 * 8 * 8, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        return self.fc(x.flatten(1))


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TinyNet()


@pytest.fixture
def batch():
    """A small batch of images in [0,1] and their labels (3 classes)."""
    torch.manual_seed(1)
    images = torch.rand(6, 1, 8, 8)
    labels = torch.randint(0, 3, (6,))
    return images, labels
