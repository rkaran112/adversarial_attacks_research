"""Experiment orchestration: evaluate attacks across an epsilon sweep."""

import os
from typing import List

import pandas as pd
import torch

from .attacks import get_attack
from .metrics import (
    accuracy,
    attack_success_rate,
    mean_l2_distortion,
    mean_linf_distortion,
)


@torch.no_grad()
def _clean_logits(model, images):
    return model(images)


def evaluate_attack(
    model,
    loader,
    attack: str,
    epsilon: float,
    device: str = "cpu",
    targeted: bool = False,
    max_batches: int = None,
    **attack_kwargs,
) -> dict:
    """Run a single attack at one epsilon over ``loader`` and aggregate metrics.

    For targeted attacks the target label is ``(true_label + 1) % num_classes``.
    Returns a dict of averaged metrics for this (attack, epsilon) cell.
    """
    attack_fn = get_attack(attack)
    model = model.to(device)
    model.eval()

    n = 0
    acc_sum = success_sum = l2_sum = linf_sum = 0.0

    for i, (images, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        num_classes = _clean_logits(model, images).shape[1]

        if targeted:
            target = (labels + 1) % num_classes
            adv = attack_fn(model, images, target, epsilon, targeted=True, **attack_kwargs)
        else:
            target = None
            adv = attack_fn(model, images, labels, epsilon, targeted=False, **attack_kwargs)

        with torch.no_grad():
            adv_logits = model(adv)

        bs = labels.size(0)
        acc_sum += accuracy(adv_logits, labels) * bs
        success_sum += attack_success_rate(adv_logits, labels, targeted, target) * bs
        l2_sum += mean_l2_distortion(adv, images) * bs
        linf_sum += mean_linf_distortion(adv, images) * bs
        n += bs

    n = max(n, 1)
    return {
        "attack": attack,
        "epsilon": epsilon,
        "targeted": targeted,
        "accuracy": acc_sum / n,
        "success_rate": success_sum / n,
        "l2": l2_sum / n,
        "linf": linf_sum / n,
        "n": n,
    }


def run_sweep(
    model,
    loader,
    attacks: List[str],
    epsilons: List[float],
    device: str = "cpu",
    targeted: bool = False,
    max_batches: int = None,
    attack_params: dict = None,
) -> pd.DataFrame:
    """Evaluate every (attack, epsilon) combination and return a results frame."""
    attack_params = attack_params or {}
    rows = []
    for attack in attacks:
        params = attack_params.get(attack, {})
        for eps in epsilons:
            rows.append(
                evaluate_attack(
                    model, loader, attack, eps,
                    device=device, targeted=targeted,
                    max_batches=max_batches, **params,
                )
            )
    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, output_dir: str, name: str = "results") -> str:
    """Write ``df`` to ``output_dir/name.csv`` and return the path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.csv")
    df.to_csv(path, index=False)
    return path
