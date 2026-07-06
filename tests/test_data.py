"""Tests for dataset metadata and loader validation.

These avoid network downloads; a full loader is only built if the dataset is
already cached locally.
"""

import os

import pytest

from src.data import get_dataloaders, DATASET_INFO


def test_dataset_info_keys():
    assert set(DATASET_INFO) == {"mnist", "cifar10"}
    assert DATASET_INFO["mnist"]["channels"] == 1
    assert DATASET_INFO["cifar10"]["channels"] == 3


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        get_dataloaders("imagenet", download=False)


@pytest.mark.skipif(
    not os.path.isdir(os.path.join("data", "MNIST")),
    reason="MNIST not downloaded; skipping loader construction",
)
def test_mnist_loader_batch_shape():
    train, _ = get_dataloaders("mnist", batch_size=8, num_workers=0, download=False)
    images, labels = next(iter(train))
    assert images.shape == (8, 1, 28, 28)
    assert labels.shape == (8,)
