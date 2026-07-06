"""Uniform dispatch layer over the individual attacks.

Every attack is exposed through a common ``(model, images, labels, epsilon,
targeted, **kwargs)`` signature so callers (the runner, tests) don't need to
special-case each one.
"""

import torch
import torch.nn as nn

from .fgsm import fgsm_attack
from .pgd import pgd_attack
from .cw import cw_attack


def _fgsm(model, images, labels, epsilon, targeted=False, **kwargs):
    return fgsm_attack(model, images, labels, epsilon, targeted=targeted)


def _pgd(model, images, labels, epsilon, targeted=False, **kwargs):
    return pgd_attack(
        model, images, labels, epsilon,
        alpha=kwargs.get("alpha", 0.01),
        steps=kwargs.get("steps", 40),
        targeted=targeted,
    )


def _cw(model, images, labels, epsilon, targeted=False, **kwargs):
    # C&W ignores epsilon; it minimizes distortion directly.
    return cw_attack(
        model, images, labels,
        c=kwargs.get("c", 1.0),
        steps=kwargs.get("steps", 50),
        lr=kwargs.get("lr", 0.01),
        targeted=targeted,
    )


ATTACKS = {"fgsm": _fgsm, "pgd": _pgd, "cw": _cw}


def get_attack(name: str):
    """Return the attack callable registered under ``name``."""
    key = name.lower()
    if key not in ATTACKS:
        raise ValueError(f"Unknown attack {name!r}; available: {sorted(ATTACKS)}")
    return ATTACKS[key]
