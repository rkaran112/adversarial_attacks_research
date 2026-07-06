"""Factory for building models by dataset name."""

import torch.nn as nn

from .lenet import LeNet5


def build_model(dataset: str, num_classes: int = 10) -> nn.Module:
    """Return an untrained model appropriate for the given dataset.

    - ``mnist``  -> :class:`LeNet5` (1-channel 28x28 input).
    - ``cifar10`` -> torchvision ``resnet18`` with its first conv/pool adapted
      for 32x32 images and the final layer resized to ``num_classes``.
    """
    dataset = dataset.lower()
    if dataset == "mnist":
        return LeNet5(num_classes=num_classes)
    if dataset == "cifar10":
        import torch.nn as _nn
        from torchvision.models import resnet18

        model = resnet18(weights=None, num_classes=num_classes)
        # Adapt the stem for small 32x32 inputs instead of 224x224.
        model.conv1 = _nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = _nn.Identity()
        return model
    raise ValueError(f"Unknown dataset: {dataset!r} (expected 'mnist' or 'cifar10')")
