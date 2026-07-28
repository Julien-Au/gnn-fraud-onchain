"""Ingestion of IBM AMLworld (HI-Small) transactions for the leakage study.

AMLworld (Altman et al., NeurIPS 2023 Datasets & Benchmarks) provides synthetic
but realistically generated transaction streams with complete ground-truth
laundering labels, distributed via Kaggle (gated by account/terms; raw files are
gitignored and never redistributed). HI-Small: ~5.08M timestamped transactions,
~0.1% labeled as laundering - an edge-level classification task.

We engineer simple per-transaction tabular features (amounts, currency/format
codes, bank/account relationships, time-of-day) plus an epoch-minute timestamp for
temporal splitting. This deliberately mirrors the paper's GBT baselines: the
leakage experiment needs a competent standard model, not a novel one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def load_amlworld(
    root: str | Path = "data/raw/amlworld",
    variant: str = "HI-Small",
    cache: str | Path = "data/processed/amlworld_hi_small.npz",
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return (features [N, D], labels [N], timestamps-in-minutes [N]).

    Cached as an npz after the first parse (the CSV is ~475 MB).
    """
    cache_path = Path(cache)
    if cache_path.exists():
        d = np.load(cache_path)
        return d["x"], d["y"], d["t"]

    df = pd.read_csv(
        Path(root) / f"{variant}_Trans.csv",
        dtype={
            "From Bank": str,
            "To Bank": str,
            "Receiving Currency": "category",
            "Payment Currency": "category",
            "Payment Format": "category",
        },
    )
    # The two "Account" columns arrive as Account / Account.1 (duplicate header).
    from_acct, to_acct = df.columns[2], df.columns[4]

    ts = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    t_min = ((ts - ts.min()).dt.total_seconds() // 60).to_numpy(dtype=np.int64)

    feats = pd.DataFrame(
        {
            "log_amount_paid": np.log1p(df["Amount Paid"].to_numpy(dtype=np.float64)),
            "log_amount_received": np.log1p(df["Amount Received"].to_numpy(dtype=np.float64)),
            "amount_mismatch": (df["Amount Paid"] != df["Amount Received"]).astype(np.int8),
            "same_currency": (
                df["Receiving Currency"].astype(str) == df["Payment Currency"].astype(str)
            ).astype(np.int8),
            "recv_currency": df["Receiving Currency"].cat.codes.astype(np.int16),
            "pay_currency": df["Payment Currency"].cat.codes.astype(np.int16),
            "pay_format": df["Payment Format"].cat.codes.astype(np.int16),
            "same_bank": (df["From Bank"] == df["To Bank"]).astype(np.int8),
            "same_account": (df[from_acct] == df[to_acct]).astype(np.int8),
            "hour": ts.dt.hour.astype(np.int8),
            "day": ts.dt.day.astype(np.int8),
        }
    )
    x = feats.to_numpy(dtype=np.float32)
    y = df["Is Laundering"].to_numpy(dtype=np.int64)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=x, y=y, t=t_min)
    return x, y, t_min
