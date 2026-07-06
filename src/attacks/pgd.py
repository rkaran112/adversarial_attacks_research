"""Projected Gradient Descent (Madry et al., 2018)."""

import torch
import torch.nn as nn


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float = 0.01,
    steps: int = 40,
    targeted: bool = False,
    random_start: bool = True,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """Craft PGD adversarial examples under an L-infinity ``epsilon`` ball.

    Iteratively takes ``steps`` gradient-sign steps of size ``alpha`` and
    projects back into the epsilon-ball around ``images`` after each step.
    """
    if epsilon == 0:
        return images.detach().clone()

    model.eval()
    original = images.clone().detach()
    adv = original.clone().detach()

    if random_start:
        adv = adv + torch.empty_like(adv).uniform_(-epsilon, epsilon)
        adv = adv.clamp(clip_min, clip_max)

    sign = -1.0 if targeted else 1.0
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = nn.functional.cross_entropy(model(adv), labels)
        grad = torch.autograd.grad(loss, adv)[0]

        adv = adv.detach() + sign * alpha * grad.sign()
        # Project into the L-inf epsilon ball, then back into valid pixel range.
        delta = (adv - original).clamp(-epsilon, epsilon)
        adv = (original + delta).clamp(clip_min, clip_max)

    return adv.detach()
