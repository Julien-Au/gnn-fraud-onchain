"""Unit test for the wallet-feature aggregation window (fast: pure pandas, but the
module imports torch, so it runs in the gnn tier)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch_geometric")

import pandas as pd  # noqa: E402

from gnn_fraud.ingestion.elliptic_pp import aggregate_wallets  # noqa: E402


def _waf() -> pd.DataFrame:
    # addr A: active pre-split (t=10, f=1.0) and post-split (t=40, f=9.0)
    # addr B: pre-split only (t=5, f=2.0)
    # addr C: post-split only (t=40, f=5.0)
    return pd.DataFrame(
        {
            "address": ["A", "A", "B", "C"],
            "Time step": [10, 40, 5, 40],
            "f": [1.0, 9.0, 2.0, 5.0],
        }
    )


def test_lifetime_leaks_post_split_activity() -> None:
    agg = aggregate_wallets(_waf(), "lifetime", max_step=34)
    assert agg.loc["A", "f"] == pytest.approx(5.0)  # (1+9)/2 - includes the future


def test_pre_split_uses_only_past_when_available() -> None:
    agg = aggregate_wallets(_waf(), "pre_split", max_step=34)
    assert agg.loc["A", "f"] == pytest.approx(1.0)  # pre-split rows only
    assert agg.loc["B", "f"] == pytest.approx(2.0)  # unchanged
    assert agg.loc["C", "f"] == pytest.approx(5.0)  # post-only address keeps its own stats


def test_unknown_window_raises() -> None:
    with pytest.raises(ValueError, match="unknown feature_window"):
        aggregate_wallets(_waf(), "nope", max_step=34)
