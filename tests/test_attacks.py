"""Tests for FGSM, PGD and C&W attacks and the dispatch registry."""

import pytest
import torch

from src.attacks import fgsm_attack, pgd_attack, cw_attack, get_attack, ATTACKS


def test_fgsm_respects_epsilon_budget(model, batch):
    images, labels = batch
    eps = 0.1
    adv = fgsm_attack(model, images, labels, epsilon=eps)
    # Perturbation is bounded by epsilon in L-inf and pixels stay valid.
    assert (adv - images).abs().max() <= eps + 1e-5
    assert adv.min() >= 0.0 and adv.max() <= 1.0
    assert adv.shape == images.shape


def test_fgsm_zero_epsilon_is_identity(model, batch):
    images, labels = batch
    adv = fgsm_attack(model, images, labels, epsilon=0.0)
    assert torch.allclose(adv, images)


def test_fgsm_increases_loss(model, batch):
    images, labels = batch
    adv = fgsm_attack(model, images, labels, epsilon=0.2)
    ce = torch.nn.functional.cross_entropy
    assert ce(model(adv), labels).item() >= ce(model(images), labels).item()


def test_pgd_respects_epsilon_ball(model, batch):
    images, labels = batch
    eps = 0.1
    adv = pgd_attack(model, images, labels, epsilon=eps, alpha=0.02, steps=10)
    assert (adv - images).abs().max() <= eps + 1e-5
    assert adv.min() >= 0.0 and adv.max() <= 1.0


def test_pgd_stronger_than_fgsm(model, batch):
    images, labels = batch
    eps = 0.15
    ce = torch.nn.functional.cross_entropy
    adv_fgsm = fgsm_attack(model, images, labels, epsilon=eps)
    adv_pgd = pgd_attack(model, images, labels, epsilon=eps, alpha=0.02, steps=20)
    # Multi-step PGD should reach at least as high a loss as one-step FGSM.
    assert ce(model(adv_pgd), labels).item() >= ce(model(adv_fgsm), labels).item() - 1e-4


def test_cw_stays_in_valid_range(model, batch):
    images, labels = batch
    adv = cw_attack(model, images, labels, c=1.0, steps=10, lr=0.05)
    assert adv.min() >= 0.0 and adv.max() <= 1.0
    assert adv.shape == images.shape


def test_targeted_fgsm_moves_toward_target(model, batch):
    images, _ = batch
    target = torch.zeros(images.size(0), dtype=torch.long)  # all class 0
    ce = torch.nn.functional.cross_entropy
    adv = fgsm_attack(model, images, target, epsilon=0.2, targeted=True)
    # Loss w.r.t. the target class should drop for a targeted attack.
    assert ce(model(adv), target).item() <= ce(model(images), target).item()


def test_registry_dispatch(model, batch):
    images, labels = batch
    for name in ["fgsm", "pgd", "cw"]:
        fn = get_attack(name)
        adv = fn(model, images, labels, epsilon=0.1)
        assert adv.shape == images.shape
    assert set(ATTACKS) == {"fgsm", "pgd", "cw"}


def test_registry_unknown_attack_raises(model, batch):
    with pytest.raises(ValueError):
        get_attack("nonexistent")
