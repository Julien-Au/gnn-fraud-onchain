# Model cards

Concise cards for every model in this project. All metrics are on the **temporal
test split** of Elliptic (transaction classification, illicit = positive), measured
and reproducible (`docs/results/*.json`). Primary metric: PR-AUC (severe class
imbalance; ~9.8% illicit among labeled). Shared limitations: strong temporal
distribution shift (a dark-market shutdown mid-sequence); no model beats the
XGBoost baseline yet.

## Common setup

- **Data**: Elliptic (PyG) for tabular/homogeneous models; Elliptic++ for the
  heterogeneous model. Real, public, cited (`data/DATA_CARD.md`).
- **Split**: built-in **temporal** train/test (early time steps -> train); a
  validation slice is carved from the latest train steps. No random split, no leak.
- **Threshold** for F1/precision/recall: chosen on validation, applied to test.
  PR-AUC / ROC-AUC are threshold-free.
- **Repro**: fixed seed (42), `uv`-locked env, one command per model.

| Model | Type | PR-AUC | ROC-AUC | F1 | Reproduce |
|---|---|---|---|---|---|
| XGBoost | tabular, supervised | 0.799 | 0.928 | 0.817 | `gnn-fraud baselines` |
| Logistic regression | tabular, supervised | 0.288 | 0.881 | 0.351 | `gnn-fraud baselines` |
| Autoencoder (USAD-style) | tabular, unsupervised | 0.038 | 0.213 | 0.122 | `gnn-fraud baselines` |
| GCN | graph, transductive | 0.294 | 0.814 | 0.436 | `gnn-fraud train-gnn` |
| GraphSAGE | graph, transductive | 0.488 | 0.877 | 0.422 | `gnn-fraud train-gnn` |
| GAT | graph, transductive | 0.332 | 0.863 | 0.379 | `gnn-fraud train-gnn` |
| EvolveGCN-O | temporal graph | 0.069 | 0.552 | 0.096 | `gnn-fraud train-temporal` |
| **Heterogeneous SAGE** | **hetero graph** | **0.586** | 0.887 | 0.538 | `gnn-fraud train-hetero` |

## Cards

### XGBoost (baseline, the bar)
- **Input**: 165/183 node features, standardized. No graph.
- **Why**: strong on imbalanced tabular; `scale_pos_weight` for the minority class.
- **Result**: PR-AUC 0.799 - the reference every graph model is measured against.
- **Limitations**: ignores relational structure; still the best here.

### Autoencoder (a la USAD)
- **Input**: standardized features; trained on **licit-only** train rows.
- **Score**: reconstruction MSE (anomaly = high error).
- **Result**: PR-AUC 0.038, ROC-AUC 0.213 (below random). **Fails**: the "normal"
  distribution shifts over time, so reconstruction error stops tracking illicitness.
  Reported and analyzed, not hidden. A USAD-style adversarial refinement is a noted
  future variant, not a claim.

### GCN / GraphSAGE / GAT (static, transductive)
- **Input**: full Elliptic graph (unknown nodes included for message passing); loss
  on labeled train nodes only.
- **Result**: 0.29 / 0.49 / 0.33 PR-AUC. Below XGBoost. SAGE > GCN > GAT (separating
  self from neighbors helps; attention does not, given mean degree ~2.3).
- **Limitations**: features already encode aggregated neighbors; per-time-step
  disconnection limits receptive field; no temporal mechanism.

### EvolveGCN-O (temporal)
- **Input**: per-time-step snapshots; GRU evolves the GCN weights across the sequence.
- **Result**: PR-AUC 0.069 under a strict far-horizon split. Diagnostic confirms it
  learns (train loss falls) but does not generalize across the temporal shift - not
  a bug. Fairer comparison (rolling protocol) is queued.

### EvolveGCN-O rolling backtest (fair shot)
- **Setup**: fuller training budget + per-time-step rolling evaluation
  (`gnn-fraud backtest-temporal`).
- **Result**: aggregate PR-AUC 0.100 (up from 0.069), but the per-step breakdown
  shows a collapse at the dark-market shutdown (steps 44-46: ~0.01). Even fairly
  evaluated, the temporal model cannot handle the regime change. Reported with the
  per-step figure so the failure is visible, not averaged away.

### Heterogeneous SAGE, address target (Elliptic++) - a different task
- **Input**: same HeteroData; target is the **address** node type; addresses split
  by first-seen time step (train 172,903 / test 92,451 labeled; 5.3% illicit).
- **Result**: PR-AUC **0.456**, F1 0.529 - detects illicit *actors* (wallets), not
  just transactions. Same model, different node type, no architecture change.
- **Reproduce**: `gnn-fraud train-hetero --target addr`.
- **Note**: not directly comparable to the tx numbers (different task/test set); the
  value is showing one schema-parameterized model serving both node types.

### Heterogeneous SAGE (Elliptic++) - the positive result
- **Input**: HeteroData with `tx` + `addr` nodes and four relations (tx-tx, addr-tx,
  tx-addr, addr-addr); `HeteroConv` with one SAGEConv per relation; features cleaned
  and z-scored. Target: same transactions, same split.
- **Result**: PR-AUC **0.586**, up from 0.488 for the tx-only GNN (+20% relative) -
  the largest graph-driven gain in the project. **The relational structure helps.**
- **Limitations**: still below XGBoost; validation PR-AUC 0.96 >> test 0.59 (temporal
  shift); address features mean-aggregated per address (documented modeling choice).
- **Relational-FM angle**: parameterized by the schema (`metadata`), not hard-coded -
  the same encoder adapts to a new node/edge-type structure with no rewrite.
