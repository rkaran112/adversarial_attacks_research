"""Metrics quantifying attack effectiveness and perturbation size."""

import torch


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 classification accuracy as a fraction in ``[0, 1]``."""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def attack_success_rate(
    adv_logits: torch.Tensor,
    labels: torch.Tensor,
    targeted: bool = False,
    target_labels: torch.Tensor = None,
) -> float:
    """Fraction of inputs for which the attack achieved its goal.

    Untargeted success = model no longer predicts the true ``labels``.
    Targeted success = model predicts the requested ``target_labels``.
    """
    preds = adv_logits.argmax(dim=1)
    if targeted:
        if target_labels is None:
            raise ValueError("target_labels required for targeted success rate")
        return (preds == target_labels).float().mean().item()
    return (preds != labels).float().mean().item()


def mean_l2_distortion(adv: torch.Tensor, clean: torch.Tensor) -> float:
    """Mean per-example L2 norm of the perturbation."""
    diff = (adv - clean).flatten(1)
    return diff.norm(p=2, dim=1).mean().item()


def mean_linf_distortion(adv: torch.Tensor, clean: torch.Tensor) -> float:
    """Mean per-example L-infinity norm of the perturbation."""
    diff = (adv - clean).flatten(1)
    return diff.abs().max(dim=1)[0].mean().item()
