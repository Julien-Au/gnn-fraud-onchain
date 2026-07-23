"""Tests for demo helpers (pure numpy, fast tier)."""

from __future__ import annotations

import numpy as np

from gnn_fraud.demo import TEST_MIN_TIMESTEP, pick_demo_timestep


def test_pick_demo_timestep_prefers_test_period_with_most_illicit() -> None:
    # time steps 1..40; illicit concentrated at step 38 within the test period.
    timesteps = np.array([1, 34, 35, 36, 38, 38, 38, 40], dtype=np.int64)
    illicit = np.array([1, 1, 0, 1, 1, 1, 1, 0], dtype=np.int64)
    # step 1 and 34 are train-period and must be ignored despite having illicit nodes.
    assert pick_demo_timestep(timesteps, illicit) == 38


def test_pick_demo_timestep_ignores_train_period() -> None:
    timesteps = np.array([10, 20, 35], dtype=np.int64)
    illicit = np.array([5, 5, 0], dtype=np.int64)  # train steps have more, but excluded
    chosen = pick_demo_timestep(timesteps, illicit)
    assert chosen >= TEST_MIN_TIMESTEP
