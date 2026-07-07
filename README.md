# Adversarial Attacks Research

Exploratory research notebook on white-box adversarial attacks (FGSM, PGD, C&W)
against image classifiers on MNIST and CIFAR-10.

> **Looking for the packaged, tested project?** The reusable implementation —
> a modular Python package with a training/evaluation CLI and a full test
> suite — now lives in its own repository:
> **[adversarial-robustness-toolkit](https://github.com/rkaran112/adversarial-robustness-toolkit)**.
> This repository holds only the original research notebook.

## Contents

- `research/Adversarial_attacks_FGSM.ipynb` — the exploratory notebook that
  implements and visualizes FGSM, PGD, and C&W attacks, sweeps over epsilon
  values, and produces accuracy plots, bar charts, a heatmap, and CSV exports.

## Setup

```bash
pip install -r requirements.txt
```

Then open the notebook in Jupyter or Colab:

```bash
jupyter notebook research/Adversarial_attacks_FGSM.ipynb
```

MNIST and CIFAR-10 are downloaded automatically on first run via
`torchvision.datasets`.

## Notebook status

This is exploratory research code, kept as-is for reference:

- The models are not trained before the attacks run (the weight-loading lines
  are commented out and there is no training loop), so the reported
  post-attack accuracy largely reflects an untrained model.
- The targeted-vs-non-targeted table uses hand-typed placeholder numbers rather
  than computed output.

The [adversarial-robustness-toolkit](https://github.com/rkaran112/adversarial-robustness-toolkit)
repository addresses these — it trains the models, implements real targeted
attacks, and verifies everything with tests.
