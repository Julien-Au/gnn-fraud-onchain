"""Experiment configuration.

Every experiment is described by a single YAML file (see ``configs/``). Loading
it into a typed dataclass - rather than passing a raw dict around - gives us
one obvious place to document each knob, cheap validation, and something for
mypy to check. Reproducibility starts here: the seed lives in the config, not
in a notebook cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """A single, reproducible experiment.

    Attributes:
        name: Human-readable experiment name, used to label output artifacts.
        dataset: Dataset key (e.g. ``"elliptic"``); resolved by the ingestion layer.
        model: Model key (e.g. ``"gcn"``, ``"sage"``, ``"gat"``); resolved by the model registry.
        seed: Global random seed. Fixed for reproducibility across runs.
        epochs: Number of training epochs.
        lr: Learning rate.
        hidden_dim: Hidden dimension of the GNN layers.
        params: Free-form model/dataset-specific extra hyperparameters.
    """

    name: str
    dataset: str
    model: str
    seed: int = 42
    epochs: int = 100
    lr: float = 1e-3
    hidden_dim: int = 64
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load an :class:`ExperimentConfig` from a YAML file.

        Raises:
            FileNotFoundError: if ``path`` does not exist.
            KeyError: if a required field is missing from the YAML.
        """
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        """Build a config from a plain dict, validating required fields."""
        required = ("name", "dataset", "model")
        missing = [key for key in required if key not in raw]
        if missing:
            raise KeyError(f"missing required config field(s): {', '.join(missing)}")

        known = {"name", "dataset", "model", "seed", "epochs", "lr", "hidden_dim"}
        params = {k: v for k, v in raw.items() if k not in known}
        return cls(
            name=str(raw["name"]),
            dataset=str(raw["dataset"]),
            model=str(raw["model"]),
            seed=int(raw.get("seed", 42)),
            epochs=int(raw.get("epochs", 100)),
            lr=float(raw.get("lr", 1e-3)),
            hidden_dim=int(raw.get("hidden_dim", 64)),
            params=params,
        )
