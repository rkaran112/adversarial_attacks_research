"""Result visualizations. Uses a non-interactive Matplotlib backend."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_accuracy_vs_epsilon(df: pd.DataFrame, out_path: str) -> str:
    """Line plot of post-attack accuracy vs. epsilon, one line per attack.

    Expects columns ``attack``, ``epsilon``, ``accuracy``. Returns ``out_path``.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for attack, group in df.groupby("attack"):
        g = group.sort_values("epsilon")
        ax.plot(g["epsilon"], g["accuracy"], marker="o", label=attack)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("accuracy under attack")
    ax.set_title("Accuracy vs. perturbation budget")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_success_heatmap(df: pd.DataFrame, out_path: str) -> str:
    """Heatmap of attack success rate over (attack, epsilon).

    Expects columns ``attack``, ``epsilon``, ``success_rate``. Returns ``out_path``.
    """
    pivot = df.pivot_table(index="attack", columns="epsilon", values="success_rate")
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_xlabel("epsilon")
    ax.set_ylabel("attack")
    ax.set_title("Attack success rate")
    fig.colorbar(im, ax=ax, label="success rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
