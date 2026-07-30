"""Imbalanced-aware binary classification metrics.

Pure numpy / scikit-learn (no torch), so these run in the fast gate and are
cheaply unit-tested. The primary metric for this project is PR-AUC (average
precision); ROC-AUC is reported alongside precisely to show how much rosier it
looks under heavy imbalance (see docs/research-log.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    """Evaluation of scores for the positive (illicit) class."""

    pr_auc: float  # average precision - the primary metric
    roc_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int

    def as_row(self) -> dict[str, float | int]:
        return {
            "pr_auc": round(self.pr_auc, 4),
            "roc_auc": round(self.roc_auc, 4),
            "f1": round(self.f1, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "threshold": round(self.threshold, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


def best_f1_threshold(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Threshold that maximizes F1 on the given (typically train) data.

    Chosen on train and then applied to test - never chosen on test.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y, s)
    if thresholds.size == 0:
        return 0.5
    # precision/recall have length len(thresholds)+1; align to thresholds.
    p = precision[:-1]
    r = recall[:-1]
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    return float(thresholds[int(np.argmax(f1))])


def evaluate_binary(
    y_true: ArrayLike, scores: ArrayLike, threshold: float | None = None
) -> BinaryMetrics:
    """Compute imbalanced-aware metrics for positive-class scores.

    ``threshold`` defaults to 0.5; pass a train-chosen threshold for the
    point-estimate metrics (F1/precision/recall/confusion).
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pr_auc = float(average_precision_score(y, s))
    roc = float(roc_auc_score(y, s))
    thr = 0.5 if threshold is None else float(threshold)
    y_pred = (s >= thr).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return BinaryMetrics(
        pr_auc=pr_auc,
        roc_auc=roc,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        threshold=thr,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )
