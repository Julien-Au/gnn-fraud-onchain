# From temporal autoencoders (USAD) to GNNs on relational data

A short narrative connecting my prior research to graph / relational learning, and
what this project actually found. Every number below is measured and reproducible
(see the [README](../README.md) and `docs/results/`); nothing is illustrative.

## Where I started: USAD (KDD 2020)

USAD detects anomalies in **multivariate time series** without labels. Two
autoencoders are trained adversarially: one learns to reconstruct normal inputs,
the other to amplify the reconstruction error of inputs the first reconstructs too
easily. The anomaly score is reconstruction error - "what the model cannot
explain." It works because (a) normal operation is approximately stationary, and
(b) the adversarial game sharpens the boundary around normal.

Two instincts carried straight into this project: **anomaly as un-explainability**,
and **structure matters** - USAD models dependence along the *time* axis; fraud on
a blockchain lives in dependence along a *graph*.

## The question this project asks

Does modeling the **graph / relational structure** of on-chain transactions help
detect fraud, over strong non-graph baselines? I built the answer bottom-up on the
Elliptic Bitcoin dataset (and its heterogeneous extension, Elliptic++), holding
myself to research-grade rigor: real public data, temporal splits with no leakage,
PR-AUC / minority-F1 (not accuracy), fixed seeds, and a CI-gated repo.

## What I found (the honest arc)

| Stage | Model | PR-AUC | Reading |
|---|---|---|---|
| Baseline | XGBoost on features | **0.799** | the bar to beat |
| USAD nod | unsupervised autoencoder | 0.038 | fails (below) |
| Static graph | GraphSAGE (tx-only) | 0.488 | graph alone < XGBoost |
| Temporal | EvolveGCN-O (strict split) | 0.069 | fails to extrapolate |
| **Relational** | **heterogeneous SAGE (tx+addr)** | **0.586** | **+20% over tx-only** |

Three findings, each reported as-is:

1. **The USAD-style autoencoder fails here (0.038, ROC-AUC 0.21 - below random).**
   Its stationarity assumption is violated: the "normal" learned on early time
   steps drifts after a dark-market shutdown, so reconstruction error stops
   tracking illicitness. This is the honest limit of transferring a time-series
   anomaly detector to a shifting graph, and it motivated everything after it.

2. **A graph alone does not beat XGBoost.** Vanilla GCN/SAGE/GAT (0.29-0.49) lose
   to XGBoost (0.80), because Elliptic's features already contain aggregated
   neighbor statistics, the graph is nearly disconnected per time step, and there
   is a hard temporal shift. A naive temporal GNN (EvolveGCN) does worse still when
   forced to extrapolate far in time - diagnosed, not hidden.

3. **Richer relational structure is what makes the graph pay off.** Adding
   address nodes and their relations (Elliptic++) lifts the GNN from 0.488 to
   **0.586 (+20% relative)** on the *same* task and split. Illicit transactions are
   betrayed by the addresses they touch - signal the tabular features and the
   tx-only graph both miss.

## The bridge to a relational foundation model

The heterogeneous model is parameterized by the **schema** (node/edge types via
`metadata`), not hard-coded to one graph: the same encoder definition adapted from
a homogeneous graph to a two-node-type, four-relation graph with no architecture
rewrite. Empirically, the richer schema gave the better model. That is the thesis a
relational foundation model scales: **one model over many relational schemas**,
where structure is a first-class input rather than flattened into features.

The line from USAD is direct: from "learn normal along a fixed time axis" to "learn
along an arbitrary, typed relational structure." The failure of the plain
autoencoder here is not a detour - it is the empirical reason to move from
reconstruction-on-features to representation-learning-on-structure.

## Honest limitations (what I would say unprompted)

- No model beats XGBoost on Elliptic yet; the graph narrows the gap, it does not
  close it. The claim is directional, and true.
- The temporal shift dominates generalization (validation PR-AUC 0.96 vs test 0.59
  for the hetero model). A fair rolling temporal protocol and explicit shift
  handling are the clear next experiments.
- Address features are mean-aggregated per address (addresses span many time steps,
  edges are timeless) - a documented modeling choice, not a free lunch.
