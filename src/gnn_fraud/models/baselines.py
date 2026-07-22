"""Non-graph baselines on Elliptic node features.

These establish an honest reference *without* graph structure, so a later GNN's
gain can be attributed to the graph rather than to the features alone:

- Logistic regression (linear, class-weight balanced) - the interpretable floor.
- XGBoost (non-linear, strong on imbalanced tabular) - the real bar to beat.
- An unsupervised autoencoder (a la USAD) - detection without illicit labels.

Protocol (see LEARNING_NOTES.md): features standardized with a scaler fit on
train only; the built-in temporal split; the F1 threshold chosen on train and
applied to test; PR-AUC is the primary metric.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from gnn_fraud.eval.metrics import BinaryMetrics, best_f1_threshold, evaluate_binary
from gnn_fraud.models.autoencoder import reconstruction_error, train_autoencoder
from gnn_fraud.train.split import labeled_temporal_split


def run_baselines(data: Data, seed: int = 42) -> dict[str, BinaryMetrics]:
    """Fit each baseline on train and evaluate on the temporal test split."""
    x_train, y_train, x_test, y_test = labeled_temporal_split(data)

    scaler = StandardScaler().fit(x_train)  # fit on train only: no leakage
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)

    results: dict[str, BinaryMetrics] = {}

    # --- Logistic regression -------------------------------------------------
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    logreg.fit(x_train_s, y_train)
    s_tr = logreg.predict_proba(x_train_s)[:, 1]
    s_te = logreg.predict_proba(x_test_s)[:, 1]
    thr = best_f1_threshold(y_train, s_tr)
    results["logreg"] = evaluate_binary(y_test, s_te, thr)

    # --- XGBoost (optional: needs the `boost` extra) -------------------------
    try:
        from xgboost import XGBClassifier

        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=(neg / pos) if pos else 1.0,
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=4,
        )
        xgb.fit(x_train_s, y_train)
        s_tr_x = xgb.predict_proba(x_train_s)[:, 1]
        s_te_x = xgb.predict_proba(x_test_s)[:, 1]
        thr_x = best_f1_threshold(y_train, s_tr_x)
        results["xgboost"] = evaluate_binary(y_test, s_te_x, thr_x)
    except ImportError:
        pass

    # --- Unsupervised autoencoder (a la USAD) --------------------------------
    x_normal = x_train_s[y_train == 0]  # train on licit only
    model = train_autoencoder(x_normal, seed=seed)
    err_tr = reconstruction_error(model, x_train_s)
    err_te = reconstruction_error(model, x_test_s)
    thr_ae = best_f1_threshold(y_train, err_tr)
    results["autoencoder"] = evaluate_binary(y_test, err_te, thr_ae)

    return results
