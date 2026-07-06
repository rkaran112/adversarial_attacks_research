"""CLI entry point: train a model, run the attack sweep, save results and plots.

Example
-------
    python main.py --config config/config.yaml
    python main.py --dataset mnist --epochs 1 --max-batches 5 --targeted
"""

import argparse
import os

from src.data import get_dataloaders
from src.models import build_model
from src.runner import run_sweep, save_results
from src.utils import load_config, set_seed, train_model, evaluate_clean
from src.visualization import plot_accuracy_vs_epsilon, plot_success_heatmap


def parse_args():
    p = argparse.ArgumentParser(description="Adversarial attack experiments")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--dataset", default=None, help="override config dataset")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--targeted", action="store_true")
    p.add_argument("--max-batches", type=int, default=None,
                   help="cap batches per train/eval loop (for quick runs)")
    p.add_argument("--no-download", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    dataset = args.dataset or cfg.dataset
    device = args.device or cfg.train.get("device", "cpu")
    epochs = args.epochs if args.epochs is not None else cfg.train.get("epochs", 1)

    set_seed(cfg.seed)

    train_loader, test_loader = get_dataloaders(
        dataset, cfg.data_dir, cfg.batch_size, cfg.num_workers,
        download=not args.no_download,
    )

    model = build_model(dataset)
    print(f"Training {dataset} model for {epochs} epoch(s) on {device}...")
    train_model(model, train_loader, epochs=epochs, lr=cfg.train.get("lr", 1e-3),
                device=device, max_batches=args.max_batches)

    clean_acc = evaluate_clean(model, test_loader, device=device, max_batches=args.max_batches)
    print(f"Clean test accuracy: {clean_acc:.3f}")

    attack_params = {"pgd": cfg.pgd, "cw": cfg.cw}
    df = run_sweep(
        model, test_loader, cfg.attacks, cfg.epsilons,
        device=device, targeted=args.targeted,
        max_batches=args.max_batches, attack_params=attack_params,
    )

    csv_path = save_results(df, cfg.output_dir)
    print(f"Saved results -> {csv_path}")
    print(df.to_string(index=False))

    os.makedirs(cfg.output_dir, exist_ok=True)
    plot_accuracy_vs_epsilon(df, os.path.join(cfg.output_dir, "accuracy_vs_epsilon.png"))
    plot_success_heatmap(df, os.path.join(cfg.output_dir, "success_heatmap.png"))
    print(f"Saved plots -> {cfg.output_dir}")


if __name__ == "__main__":
    main()
