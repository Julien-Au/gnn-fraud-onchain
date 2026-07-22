"""Tests for the experiment config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from gnn_fraud.config import ExperimentConfig


def test_from_dict_minimal_uses_defaults() -> None:
    cfg = ExperimentConfig.from_dict({"name": "exp", "dataset": "elliptic", "model": "gcn"})
    assert cfg.name == "exp"
    assert cfg.seed == 42
    assert cfg.epochs == 100
    assert cfg.lr == pytest.approx(1e-3)
    assert cfg.hidden_dim == 64
    assert cfg.params == {}


def test_from_dict_collects_extra_params() -> None:
    cfg = ExperimentConfig.from_dict(
        {"name": "exp", "dataset": "elliptic", "model": "gat", "heads": 4, "dropout": 0.5}
    )
    assert cfg.params == {"heads": 4, "dropout": 0.5}


def test_from_dict_missing_required_raises() -> None:
    with pytest.raises(KeyError, match="dataset"):
        ExperimentConfig.from_dict({"name": "exp", "model": "gcn"})


def test_from_yaml_roundtrip(tmp_path: Path) -> None:
    yaml_path = tmp_path / "exp.yaml"
    yaml_path.write_text(
        "name: gcn-elliptic\ndataset: elliptic\nmodel: gcn\nseed: 7\nhidden_dim: 128\n",
        encoding="utf-8",
    )
    cfg = ExperimentConfig.from_yaml(yaml_path)
    assert cfg.name == "gcn-elliptic"
    assert cfg.seed == 7
    assert cfg.hidden_dim == 128


def test_config_is_frozen() -> None:
    cfg = ExperimentConfig.from_dict({"name": "exp", "dataset": "elliptic", "model": "gcn"})
    with pytest.raises((AttributeError, TypeError)):
        cfg.seed = 1  # type: ignore[misc]
