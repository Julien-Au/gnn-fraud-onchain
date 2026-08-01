"""Ingestion of Elliptic++ into a heterogeneous graph (HeteroData).

Elliptic++ (Elmougy & Liu, KDD '23) adds wallet **addresses** on top of Elliptic's
transactions, giving a genuinely heterogeneous graph:

- node types: ``tx`` (transactions) and ``addr`` (addresses)
- edge types: tx->tx, addr->tx, tx->addr, addr->addr

Design choices (documented in data/DATA_CARD.md):

- The **target is selectable** (``tx`` or ``addr``): transaction classification
  keeps the cleanest temporal split (each transaction has a single time step),
  while address classification supervises the ``addr`` nodes (split by first-seen
  time step, labels are per-entity). The non-target type serves as context.
- Wallet features are given per address-*time step* (1.27M rows) but the edge lists
  reference addresses by string only, so a node is a **unique address**; we
  aggregate its features by mean over time steps.
- Edges referencing an id absent from the feature tables are dropped (logged).

Parsing ~2 GB of CSVs is done once and cached to ``data/processed`` as a ``.pt``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch_geometric.data import HeteroData

# Elliptic++ class encoding: 1 = illicit, 2 = licit, 3 = unknown.
PP_ILLICIT, PP_LICIT, PP_UNKNOWN = 1, 2, 3
TRAIN_MAX_TIMESTEP = 34


def _index_map(ids: NDArray[Any]) -> dict[object, int]:
    return {v: i for i, v in enumerate(ids)}


def _clean_features(feat: NDArray[Any], fit_mask: NDArray[Any] | None = None) -> NDArray[Any]:
    """Replace non-finite values with 0 and z-score each column.

    Wallet features mix wildly different scales (block numbers ~4e5, small counts)
    and contain NaNs; without this the GNN produces NaN predictions. By default the
    scaling statistics use all nodes (feature-only, no labels - the transductive
    convention used by our published numbers). Pass ``fit_mask`` to fit the
    statistics on train-period nodes only (the deployment-honest convention).
    """
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    fit = feat[fit_mask] if fit_mask is not None else feat
    mean = fit.mean(axis=0, keepdims=True)
    std = fit.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    scaled: NDArray[Any] = ((feat - mean) / std).astype(np.float32)
    return scaled


def aggregate_wallets(waf: pd.DataFrame, feature_window: str, max_step: int) -> pd.DataFrame:
    """Aggregate per-address-timestep wallet features to one row per address.

    ``feature_window='lifetime'``: mean over the address's whole lifetime (the
    original convention; leaks post-split activity into train-node features).
    ``feature_window='pre_split'``: mean over time steps <= ``max_step`` when the
    address has such rows; addresses first seen after the split keep their own
    (post-split) statistics, which is legitimate - they are test-period nodes.
    """
    body = waf.drop(columns=["Time step"])
    agg_all: pd.DataFrame = body.groupby("address").mean(numeric_only=True)
    if feature_window == "lifetime":
        return agg_all
    if feature_window != "pre_split":
        raise ValueError(f"unknown feature_window '{feature_window}'")
    pre = waf[waf["Time step"] <= max_step].drop(columns=["Time step"])
    agg_pre: pd.DataFrame = pre.groupby("address").mean(numeric_only=True)
    out: pd.DataFrame = agg_all.copy()
    out.loc[agg_pre.index] = agg_pre
    return out


def _map_edges(
    df: pd.DataFrame,
    src_col: str,
    dst_col: str,
    src_index: dict[object, int],
    dst_index: dict[object, int],
) -> torch.Tensor:
    """Map a two-column edge dataframe to a [2, E] index tensor, dropping unknowns."""
    src = df[src_col].map(src_index)
    dst = df[dst_col].map(dst_index)
    keep = src.notna() & dst.notna()
    edge = np.stack([src[keep].to_numpy(), dst[keep].to_numpy()]).astype(np.int64)
    return torch.from_numpy(edge)


def build_hetero(
    root: str | Path = "data/raw/elliptic_pp",
    cache: str | Path = "data/processed/elliptic_pp.pt",
    feature_window: str = "lifetime",
) -> HeteroData:
    """Build (and cache) the Elliptic++ heterogeneous graph.

    ``feature_window='pre_split'`` builds the deployment-honest variant: wallet
    features aggregated over pre-split activity only, and z-scoring statistics fit
    on train-period nodes only. ``'lifetime'`` reproduces the original convention.
    """
    cache_path = Path(cache)
    if feature_window != "lifetime":
        cache_path = cache_path.with_name(cache_path.stem + f"_{feature_window}.pt")
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    root = Path(root)
    honest = feature_window == "pre_split"

    # --- Transaction nodes ---------------------------------------------------
    txf = pd.read_csv(root / "txs_features.csv")
    tx_ids = txf["txId"].to_numpy()
    tx_time = txf["Time step"].to_numpy().astype(np.int64)
    tx_raw = txf.drop(columns=["txId", "Time step"]).to_numpy(dtype=np.float32)
    tx_feat = _clean_features(tx_raw, fit_mask=(tx_time <= TRAIN_MAX_TIMESTEP) if honest else None)
    tx_index = _index_map(tx_ids)

    txc = pd.read_csv(root / "txs_classes.csv").set_index("txId")["class"]
    tx_cls = txc.reindex(tx_ids).to_numpy()  # 1 illicit, 2 licit, 3 unknown
    tx_y = np.full(len(tx_ids), -1, dtype=np.int64)
    tx_y[tx_cls == PP_ILLICIT] = 1
    tx_y[tx_cls == PP_LICIT] = 0

    # --- Address nodes (features aggregated per unique address) --------------
    waf = pd.read_csv(root / "wallets_features.csv")
    first_seen = waf.groupby("address")["Time step"].min()  # split by first appearance
    addr_agg = aggregate_wallets(waf, feature_window, TRAIN_MAX_TIMESTEP)
    addr_ids = addr_agg.index.to_numpy()
    addr_time = first_seen.reindex(addr_ids).to_numpy().astype(np.int64)
    addr_feat = _clean_features(
        addr_agg.to_numpy(dtype=np.float32),
        fit_mask=(addr_time <= TRAIN_MAX_TIMESTEP) if honest else None,
    )
    addr_index = _index_map(addr_ids)

    wac = pd.read_csv(root / "wallets_classes.csv").set_index("address")["class"]
    addr_cls = wac.reindex(addr_ids).to_numpy()
    addr_y = np.full(len(addr_ids), -1, dtype=np.int64)
    addr_y[addr_cls == PP_ILLICIT] = 1
    addr_y[addr_cls == PP_LICIT] = 0

    # --- Edges ---------------------------------------------------------------
    tx_tx = _map_edges(pd.read_csv(root / "txs_edgelist.csv"), "txId1", "txId2", tx_index, tx_index)
    addr_tx = _map_edges(
        pd.read_csv(root / "AddrTx_edgelist.csv"), "input_address", "txId", addr_index, tx_index
    )
    tx_addr = _map_edges(
        pd.read_csv(root / "TxAddr_edgelist.csv"), "txId", "output_address", tx_index, addr_index
    )
    addr_addr = _map_edges(
        pd.read_csv(root / "AddrAddr_edgelist.csv"),
        "input_address",
        "output_address",
        addr_index,
        addr_index,
    )

    # --- Assemble HeteroData -------------------------------------------------
    data = HeteroData()
    data["tx"].x = torch.from_numpy(tx_feat)
    data["tx"].y = torch.from_numpy(tx_y)
    data["tx"].time = torch.from_numpy(tx_time)
    labeled = torch.from_numpy(tx_y >= 0)
    tt = torch.from_numpy(tx_time)
    data["tx"].train_mask = labeled & (tt <= TRAIN_MAX_TIMESTEP)
    data["tx"].test_mask = labeled & (tt > TRAIN_MAX_TIMESTEP)
    data["addr"].x = torch.from_numpy(addr_feat)
    data["addr"].y = torch.from_numpy(addr_y)
    data["addr"].time = torch.from_numpy(addr_time)
    addr_labeled = torch.from_numpy(addr_y >= 0)
    att = torch.from_numpy(addr_time)
    data["addr"].train_mask = addr_labeled & (att <= TRAIN_MAX_TIMESTEP)
    data["addr"].test_mask = addr_labeled & (att > TRAIN_MAX_TIMESTEP)

    data["tx", "to", "tx"].edge_index = tx_tx
    data["addr", "to", "tx"].edge_index = addr_tx
    data["tx", "to", "addr"].edge_index = tx_addr
    data["addr", "to", "addr"].edge_index = addr_addr

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    return data
