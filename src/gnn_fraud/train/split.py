"""Extract the built-in temporal train/test split as plain arrays.

PyG's ``train_mask`` / ``test_mask`` already select only the labeled nodes and
already encode the temporal split (early time steps -> train). We surface them as
numpy arrays for the non-graph baselines; we never re-randomize them.
"""

from __future__ import annotations

from typing import Any

from numpy.typing import NDArray
from torch_geometric.data import Data


def labeled_temporal_split(
    data: Data,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return (x_train, y_train, x_test, y_test) for labeled nodes only.

    Labels are 0 (licit) / 1 (illicit); unknown nodes are excluded by the masks.
    """
    tr = data.train_mask
    te = data.test_mask
    x_train = data.x[tr].cpu().numpy()
    y_train = data.y[tr].cpu().numpy().astype(int)
    x_test = data.x[te].cpu().numpy()
    y_test = data.y[te].cpu().numpy().astype(int)
    return x_train, y_train, x_test, y_test
