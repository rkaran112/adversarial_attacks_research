"""Tests for evaluation metrics."""

import torch

from src.metrics import (
    accuracy,
    attack_success_rate,
    mean_l2_distortion,
    mean_linf_distortion,
)


def test_accuracy_perfect_and_zero():
    logits = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    assert accuracy(logits, torch.tensor([0, 1])) == 1.0
    assert accuracy(logits, torch.tensor([1, 0])) == 0.0


def test_untargeted_success_rate():
    # Predictions: [1, 1]; true labels [0, 1] -> one flipped, one not.
    logits = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    rate = attack_success_rate(logits, torch.tensor([0, 1]), targeted=False)
    assert rate == 0.5


def test_targeted_success_rate():
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    targets = torch.tensor([1, 1])  # only first matches
    rate = attack_success_rate(logits, torch.tensor([0, 0]), targeted=True, target_labels=targets)
    assert rate == 0.5


def test_distortion_norms():
    clean = torch.zeros(2, 1, 2, 2)
    adv = clean.clone()
    adv[0, 0, 0, 0] = 0.5
    assert mean_linf_distortion(adv, clean) == 0.25  # (0.5 + 0) / 2
    # L2: first example norm 0.5, second 0 -> mean 0.25
    assert abs(mean_l2_distortion(adv, clean) - 0.25) < 1e-6
