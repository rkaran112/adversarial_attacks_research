"""Tests for config loading, seeding, and training loops."""

import torch

from src.utils import load_config, set_seed, train_model, evaluate_clean
from src.utils.config import Config


def test_load_config_defaults(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("dataset: cifar10\nbatch_size: 64\n")
    cfg = load_config(str(cfg_file))
    assert isinstance(cfg, Config)
    assert cfg.dataset == "cifar10"
    assert cfg.batch_size == 64
    # Unspecified fields fall back to defaults.
    assert cfg.seed == 0


def test_set_seed_is_reproducible():
    set_seed(42)
    a = torch.rand(5)
    set_seed(42)
    b = torch.rand(5)
    assert torch.allclose(a, b)


def test_train_reduces_loss(model, batch):
    images, labels = batch
    loader = [(images, labels)]
    ce = torch.nn.functional.cross_entropy
    before = ce(model(images), labels).item()
    train_model(model, loader, epochs=30, lr=0.05, device="cpu")
    after = ce(model(images), labels).item()
    assert after < before


def test_evaluate_clean_returns_fraction(model, batch):
    images, labels = batch
    acc = evaluate_clean(model, [(images, labels)], device="cpu")
    assert 0.0 <= acc <= 1.0
