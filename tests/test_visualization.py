"""Tests that plotting helpers produce output files without a display."""

import os

import pandas as pd

from src.visualization import plot_accuracy_vs_epsilon, plot_success_heatmap


def _sample_df():
    return pd.DataFrame(
        [
            {"attack": "fgsm", "epsilon": 0.0, "accuracy": 0.9, "success_rate": 0.0},
            {"attack": "fgsm", "epsilon": 0.1, "accuracy": 0.4, "success_rate": 0.6},
            {"attack": "pgd", "epsilon": 0.0, "accuracy": 0.9, "success_rate": 0.0},
            {"attack": "pgd", "epsilon": 0.1, "accuracy": 0.2, "success_rate": 0.8},
        ]
    )


def test_plot_accuracy_vs_epsilon(tmp_path):
    out = str(tmp_path / "acc.png")
    path = plot_accuracy_vs_epsilon(_sample_df(), out)
    assert os.path.exists(path) and os.path.getsize(path) > 0


def test_plot_success_heatmap(tmp_path):
    out = str(tmp_path / "heat.png")
    path = plot_success_heatmap(_sample_df(), out)
    assert os.path.exists(path) and os.path.getsize(path) > 0
