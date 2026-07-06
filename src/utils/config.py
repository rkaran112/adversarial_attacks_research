"""Configuration loading from YAML with attribute-style access."""

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List

import yaml


@dataclass
class Config:
    """Typed view over the experiment configuration."""

    dataset: str = "mnist"
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 2
    train: Dict[str, Any] = field(default_factory=lambda: {"epochs": 1, "lr": 1e-3, "device": "cpu"})
    attacks: List[str] = field(default_factory=lambda: ["fgsm", "pgd", "cw"])
    epsilons: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.3])
    pgd: Dict[str, Any] = field(default_factory=lambda: {"alpha": 0.01, "steps": 40})
    cw: Dict[str, Any] = field(default_factory=lambda: {"c": 1.0, "steps": 50, "lr": 0.01})
    seed: int = 0
    output_dir: str = "./outputs"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: str) -> Config:
    """Load a :class:`Config` from a YAML file."""
    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    return Config.from_dict(data)
