"""Tests for the experiment runner and result aggregation."""

import pandas as pd

from src.runner import evaluate_attack, run_sweep, save_results


def test_evaluate_attack_metrics(model, batch):
    images, labels = batch
    loader = [(images, labels)]
    row = evaluate_attack(model, loader, "fgsm", epsilon=0.1, device="cpu")
    assert row["attack"] == "fgsm"
    assert row["epsilon"] == 0.1
    assert 0.0 <= row["accuracy"] <= 1.0
    assert 0.0 <= row["success_rate"] <= 1.0
    assert row["n"] == labels.size(0)


def test_run_sweep_shape(model, batch):
    images, labels = batch
    loader = [(images, labels)]
    df = run_sweep(model, loader, attacks=["fgsm", "pgd"], epsilons=[0.0, 0.1], device="cpu")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4  # 2 attacks x 2 epsilons
    assert set(df["attack"]) == {"fgsm", "pgd"}


def test_zero_epsilon_is_zero_distortion(model, batch):
    images, labels = batch
    loader = [(images, labels)]
    row = evaluate_attack(model, loader, "fgsm", epsilon=0.0, device="cpu")
    # No perturbation -> zero distortion; adversarial accuracy equals clean.
    from src.utils import evaluate_clean

    assert row["linf"] == 0.0
    assert row["l2"] == 0.0
    assert abs(row["accuracy"] - evaluate_clean(model, loader, device="cpu")) < 1e-6


def test_targeted_sweep_runs(model, batch):
    images, labels = batch
    loader = [(images, labels)]
    df = run_sweep(model, loader, attacks=["fgsm"], epsilons=[0.2], device="cpu", targeted=True)
    assert df.iloc[0]["targeted"]


def test_save_results(tmp_path, model, batch):
    images, labels = batch
    df = run_sweep(model, [(images, labels)], attacks=["fgsm"], epsilons=[0.1], device="cpu")
    path = save_results(df, str(tmp_path), name="out")
    assert path.endswith("out.csv")
    reloaded = pd.read_csv(path)
    assert len(reloaded) == 1
