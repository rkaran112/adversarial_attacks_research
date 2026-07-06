"""Tests for model architectures and the factory."""

import pytest
import torch

from src.models import LeNet5, build_model


def test_lenet_output_shape():
    model = LeNet5()
    out = model(torch.rand(4, 1, 28, 28))
    assert out.shape == (4, 10)


def test_build_mnist_returns_lenet():
    model = build_model("mnist")
    assert isinstance(model, LeNet5)
    assert model(torch.rand(2, 1, 28, 28)).shape == (2, 10)


def test_build_cifar10_shape():
    model = build_model("cifar10")
    out = model(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_build_unknown_dataset_raises():
    with pytest.raises(ValueError):
        build_model("imagenet")
