# Adversarial Attacks Research

Research notebook exploring adversarial attacks (FGSM, PGD, C&W) against image classifiers on MNIST and CIFAR-10.

## What it does

`src/Adversarial_attacks_FGSM.ipynb` implements and evaluates three white-box adversarial attacks:

- **FGSM** (Fast Gradient Sign Method)
- **PGD** (Projected Gradient Descent)
- **C&W** (Carlini-Wagner)

It loads MNIST (via a `LeNet5` model) and CIFAR-10 (via a `torchvision` `resnet50`), runs each attack at a few epsilon values, prints post-attack accuracy, and produces summary plots (accuracy vs. epsilon, bar charts, a heatmap) and CSV exports of the results.

## Tech stack

- Python (Jupyter notebook)
- PyTorch / torchvision
- pandas, seaborn, matplotlib, tabulate

## Setup

1. Install the packages the notebook imports:
   ```
   pip install -r requirements.txt
   ```
2. Open `src/Adversarial_attacks_FGSM.ipynb` in Jupyter/Colab. MNIST and CIFAR-10 are downloaded automatically on first run via `torchvision.datasets`.

## Usage

Run the notebook top to bottom. It will:
1. Download MNIST/CIFAR-10 and build `DataLoader`s.
2. Instantiate `LeNet5` (for MNIST) and a `resnet50` with its final layer swapped in for 10 classes (for CIFAR-10).
3. Run `evaluate_attack(loader, model, attack="FGSM"|"PGD"|"CW", epsilon=...)` across a few epsilon values and print/plot accuracy under attack.

## Status

**Work in progress.** Specifically, based on reading the actual notebook:

- Neither model is trained before the attacks run — the lines that would load trained weights (`lenet.load_state_dict(...)`, `resnet.load_state_dict(...)`) are commented out, and there is no training loop. As a result, the reported "post-attack" accuracy (~10%, i.e. random-chance for 10 classes) mostly reflects an untrained model rather than the effect of the attack.
- The "targeted vs. non-targeted success rate" table/plot in the results section uses hand-typed placeholder numbers (`success_rate_data` is a literal dict in the code), not output computed from an actual targeted-attack implementation — no targeted attack is implemented in the notebook.
- `src/implementation.ipynb` is present but is an empty file (0 bytes), so it can't currently be opened as a notebook.
- No tests or CLI/script entry point exist — this is a single exploratory notebook, not a packaged project.

Next steps to make this "complete": train the models (or load real pretrained weights) before running the attacks, implement an actual targeted-attack path instead of hard-coded example numbers, and either fill in or remove `src/implementation.ipynb`.
