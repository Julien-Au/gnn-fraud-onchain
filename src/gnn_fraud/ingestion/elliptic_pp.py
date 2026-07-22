"""Ingestion of Elliptic++ into a heterogeneous graph (HeteroData).

Elliptic++ (Elmougy & Liu, KDD '23) adds wallet **addresses** on top of Elliptic's
transactions, giving a genuinely heterogeneous graph:

- node types: ``tx`` (transactions) and ``addr`` (addresses)
- edge types: tx->tx, addr->tx, tx->addr, addr->addr

Design choices (documented in data/DATA_CARD.md):

- The **target is transaction classification** (illicit vs licit), so the temporal
  split stays clean: each transaction has a single time step, whereas an address
  spans several. Addresses are context nodes (features only, no supervision here).
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


def _clean_features(feat: NDArray[Any]) -> NDArray[Any]:
    """Replace non-finite values with 0 and z-score each column.

    Wallet features mix wildly different scales (block numbers ~4e5, small counts)
    and contain NaNs; without this the GNN produces NaN predictions. Scaling uses
    all nodes (feature-only, no labels), standard in transductive settings.
    """
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    mean = feat.mean(axis=0, keepdims=True)
    std = feat.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    scaled: NDArray[Any] = ((feat - mean) / std).astype(np.float32)
    return scaled


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
) -> HeteroData:
    """Build (and cache) the Elliptic++ heterogeneous graph."""
    cache_path = Path(cache)
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)

    root = Path(root)

    # --- Transaction nodes ---------------------------------------------------
    txf = pd.read_csv(root / "txs_features.csv")
    tx_ids = txf["txId"].to_numpy()
    tx_time = txf["Time step"].to_numpy().astype(np.int64)
    tx_feat = _clean_features(txf.drop(columns=["txId", "Time step"]).to_numpy(dtype=np.float32))
    tx_index = _index_map(tx_ids)

    txc = pd.read_csv(root / "txs_classes.csv").set_index("txId")["class"]
    tx_cls = txc.reindex(tx_ids).to_numpy()  # 1 illicit, 2 licit, 3 unknown
    tx_y = np.full(len(tx_ids), -1, dtype=np.int64)
    tx_y[tx_cls == PP_ILLICIT] = 1
    tx_y[tx_cls == PP_LICIT] = 0

    # --- Address nodes (features aggregated per unique address) --------------
    waf = pd.read_csv(root / "wallets_features.csv")
    waf = waf.drop(columns=["Time step"])
    addr_agg = waf.groupby("address").mean(numeric_only=True)
    addr_ids = addr_agg.index.to_numpy()
    addr_feat = _clean_features(addr_agg.to_numpy(dtype=np.float32))
    addr_index = _index_map(addr_ids)

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

    data["tx", "to", "tx"].edge_index = tx_tx
    data["addr", "to", "tx"].edge_index = addr_tx
    data["tx", "to", "addr"].edge_index = tx_addr
    data["addr", "to", "addr"].edge_index = addr_addr

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    return data
