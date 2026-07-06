# Adversarial Attacks Research

White-box adversarial attacks (FGSM, PGD, C&W) against image classifiers on
MNIST and CIFAR-10, packaged as a tested, modular Python project.

The original exploratory notebook lives in [`research/`](research/); the
reusable, tested implementation lives in [`src/`](src/) with a CLI entry point
in [`main.py`](main.py).

## What it does

Trains a classifier (LeNet-5 for MNIST, an adapted ResNet-18 for CIFAR-10),
then evaluates three white-box attacks across an epsilon sweep and reports
post-attack accuracy, attack success rate, and perturbation size (L2 / L∞):

- **FGSM** — Fast Gradient Sign Method (single-step, L∞).
- **PGD** — Projected Gradient Descent (multi-step, L∞).
- **C&W** — Carlini–Wagner L2 (distortion-minimizing).

All three support both untargeted and targeted modes. Results are exported to
CSV and rendered as an accuracy-vs-epsilon line plot and a success-rate heatmap.

## Project layout

```
src/
  attacks/        FGSM, PGD, C&W + a uniform dispatch registry
  data/           MNIST / CIFAR-10 DataLoaders (pixels kept in [0,1])
  models/         LeNet-5, ResNet-18 factory
  metrics/        accuracy, attack success rate, L2 / L∞ distortion
  utils/          config loading, seeding, train / eval loops
  visualization/  accuracy-vs-epsilon plot, success heatmap
  runner.py       orchestrates the (attack × epsilon) sweep
main.py           CLI: train -> sweep -> save CSV + plots
config/config.yaml default experiment configuration
tests/            pytest suite (fast, synthetic, no dataset download)
research/         original exploratory notebook
```

## Setup

```bash
pip install -r requirements.txt      # or: pip install -e .
```

PyTorch is the one large dependency; install a build matching your platform
from https://pytorch.org if the default wheel doesn't suit you.

## Usage

Run the full pipeline with the default config:

```bash
python main.py --config config/config.yaml
```

Useful overrides:

```bash
# Quick smoke run: cap batches per train/eval loop
python main.py --dataset mnist --epochs 1 --max-batches 40

# Targeted attacks (target = (true_label + 1) % num_classes)
python main.py --dataset mnist --targeted
```

Outputs (CSV + PNG plots) are written to `outputs/` (gitignored).

### As a library

```python
from src.models import build_model
from src.attacks import fgsm_attack
from src.data import get_dataloaders

model = build_model("mnist")
adv = fgsm_attack(model, images, labels, epsilon=0.1)          # untargeted
adv = fgsm_attack(model, images, targets, epsilon=0.1, targeted=True)
```

## Tests

```bash
pytest -q
```

The suite covers every module (attacks, models, metrics, utils, runner,
visualization, data) using tiny synthetic tensors and a small in-memory model,
so it runs in seconds and requires no dataset download.

## Example results (MNIST, LeNet-5)

Attacks degrade a trained model as the perturbation budget grows; PGD is
consistently stronger than single-step FGSM, while C&W finds misclassifications
with very small L2 distortion rather than sweeping a budget:

| attack | ε=0.0 | ε=0.1 | ε=0.2 | ε=0.3 |
|--------|-------|-------|-------|-------|
| FGSM   | 0.71  | 0.39  | 0.10  | 0.04  |
| PGD    | 0.71  | 0.31  | 0.06  | 0.02  |

(accuracy under attack; exact numbers depend on training length and seed.)
