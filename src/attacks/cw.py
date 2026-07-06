"""Carlini & Wagner L2 attack (Carlini & Wagner, 2017)."""

import torch
import torch.nn as nn


def cw_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    c: float = 1.0,
    kappa: float = 0.0,
    steps: int = 50,
    lr: float = 0.01,
    targeted: bool = False,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    **_ignored,
) -> torch.Tensor:
    """Craft C&W L2 adversarial examples.

    Optimizes in ``tanh`` space so perturbed inputs stay in ``[clip_min,
    clip_max]``, minimizing an L2 distortion term plus ``c`` times a margin
    loss (``f6`` from the paper). ``epsilon`` is intentionally not a parameter;
    the attack minimizes distortion rather than working under a fixed budget.
    """
    model.eval()
    span = clip_max - clip_min

    # Map images into tanh space: x = clip_min + span * (tanh(w)+1)/2.
    normalized = ((images - clip_min) / span).clamp(1e-6, 1 - 1e-6)
    w = torch.atanh(normalized * 2 - 1).detach()
    w.requires_grad_(True)

    optimizer = torch.optim.Adam([w], lr=lr)
    num_classes = model(images).shape[1]
    one_hot = nn.functional.one_hot(labels, num_classes).float()

    best_adv = images.clone().detach()

    for _ in range(steps):
        adv = clip_min + span * (torch.tanh(w) + 1) / 2
        logits = model(adv)

        real = (one_hot * logits).sum(dim=1)
        other = ((1 - one_hot) * logits - one_hot * 1e4).max(dim=1)[0]

        if targeted:
            # Push the target logit above every other logit.
            f = torch.clamp(other - real + kappa, min=0)
        else:
            # Push the true logit below the best other logit.
            f = torch.clamp(real - other + kappa, min=0)

        l2 = ((adv - images) ** 2).flatten(1).sum(dim=1)
        loss = (l2 + c * f).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        best_adv = adv.detach()

    return best_adv.clamp(clip_min, clip_max)
