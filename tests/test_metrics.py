"""Tests for the imbalanced-aware metrics (fast tier, no torch)."""

from __future__ import annotations

import numpy as np

from gnn_fraud.eval.metrics import best_f1_threshold, evaluate_binary


def test_perfect_separation() -> None:
    y = [0, 0, 0, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.9, 0.95]
    m = evaluate_binary(y, scores, threshold=0.5)
    assert m.pr_auc == 1.0
    assert m.roc_auc == 1.0
    assert m.f1 == 1.0
    assert (m.tp, m.fp, m.fn, m.tn) == (2, 0, 0, 3)


def test_all_negative_predictor_has_high_roc_but_zero_f1_intuition() -> None:
    # 98 negatives, 2 positives; a constant low score misses every positive.
    y = np.array([0] * 98 + [1] * 2)
    scores = np.full(100, 0.01)
    m = evaluate_binary(y, scores, threshold=0.5)
    # Predicting all-negative: no true positives -> F1 is zero...
    assert m.f1 == 0.0
    assert m.tp == 0 and m.fn == 2
    # ...even though "accuracy" would be 98%. This is why we never report accuracy.
    accuracy = (m.tn + m.tp) / (m.tn + m.tp + m.fn + m.fp)
    assert accuracy == 0.98


def test_pr_auc_lower_than_roc_under_imbalance() -> None:
    rng = np.random.default_rng(0)
    y = np.array([0] * 950 + [1] * 50)
    # Weakly informative scores: positives shifted up a little.
    scores = rng.normal(0.0, 1.0, size=1000)
    scores[y == 1] += 1.0
    m = evaluate_binary(y, scores)
    # Under heavy imbalance ROC-AUC looks rosier than PR-AUC.
    assert m.roc_auc > m.pr_auc


def test_best_f1_threshold_recovers_a_good_cut() -> None:
    y = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.6, 0.9]
    thr = best_f1_threshold(y, scores)
    m = evaluate_binary(y, scores, threshold=thr)
    assert m.f1 == 1.0
