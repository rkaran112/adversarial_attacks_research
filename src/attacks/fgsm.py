"""Fast Gradient Sign Method (Goodfellow et al., 2015)."""

import torch
import torch.nn as nn


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    targeted: bool = False,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """Craft FGSM adversarial examples.

    Perturbs each input a single step of size ``epsilon`` along the sign of the
    loss gradient. For a targeted attack, ``labels`` are the desired target
    classes and the step descends the loss toward them.
    """
    if epsilon == 0:
        return images.detach().clone()

    model.eval()
    adv = images.clone().detach().requires_grad_(True)
    loss = nn.functional.cross_entropy(model(adv), labels)
    grad = torch.autograd.grad(loss, adv)[0]

    # Untargeted: ascend the loss. Targeted: descend toward the target label.
    sign = -1.0 if targeted else 1.0
    adv = adv.detach() + sign * epsilon * grad.sign()
    return adv.clamp(clip_min, clip_max)
