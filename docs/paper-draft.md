# The Elliptic "SOTA" is a Leakage Mirage: A Leakage-Free Re-Evaluation of Graph Fraud Detection Under Temporal Distribution Shift

*Working draft. All numbers are measured and reproducible from this repository
(`docs/research-log.md`, `docs/results/`). This is an honest reality-check /
evaluation contribution, not a claim of a new state-of-the-art model.*

## Abstract

The Elliptic Bitcoin dataset and its heterogeneous extension Elliptic++ are standard
benchmarks for graph-based fraud detection, and a stream of recent work reports
near-perfect illicit-detection performance (F1 up to ~0.98). We show these numbers are
largely an artifact of evaluation leakage. Using a single codebase, we train the *same*
models under (a) an honest temporal split (train on early time steps, test on later
ones) and (b) a random split, and find that the random split inflates illicit-class
PR-AUC by +0.20 to +0.52 across four models and two tasks: under the random split a
vanilla GraphSAGE reaches F1 0.86 and XGBoost reaches PR-AUC 0.987 / F1 0.955 - squarely
in the reported "SOTA" range - while the identical models score far lower under temporal
evaluation. The inflation generalizes from the Elliptic transaction task to the
Elliptic++ address task (+0.52) and is robust across seeds. Critically, a cross-domain
control - DGraph-Fin, a temporally *stable* fintech graph - shows *no* inflation
(+0.002), isolating the cause: the inflation is driven by temporal distribution shift,
not by random splitting per se, which sharpens the warning for shifting benchmarks like
Elliptic. Under the only comparable protocol - illicit-class F1 with a temporal split -
gradient-boosted trees still beat graph neural networks by 10+ points, reproducing the
original 2019 benchmark. We further characterize the field's genuine open problem, the
post-time-step-43 "dark-market shutdown" distribution shift, under which per-window
PR-AUC collapses to ~0.01 and which no surveyed method closes, and we report novel but
negative attempts to address it with a USAD-style adversarial graph autoencoder and a
supervised domain-adversarial GNN (four variants, all failing). We recommend that graph-fraud papers report
illicit-class F1 / PR-AUC under a strict temporal split and treat the post-shift window
as the real benchmark.

## 1. Introduction

Detecting illicit activity in blockchain transaction graphs is a canonical application of
graph machine learning. The Elliptic dataset (203,769 Bitcoin transactions across 49 time
steps, ~2% illicit) and Elliptic++ (adding 822,942 wallet addresses) are the de-facto
benchmarks. Recent papers report illicit-detection F1 approaching 0.98, suggesting the
problem is nearly solved. We argue this impression is false and traces to two evaluation
choices: (i) random rather than temporal train/test splits, which leak future information
into training, and (ii) reporting weighted/overall F1 (dominated by the ~98% licit class)
rather than illicit-class F1. Our contribution is a leakage-free re-evaluation that
quantifies the inflation, reproduces the honest state of the art, characterizes the real
open problem, and documents a novel-but-negative method attempt.

## 2. Background and related work

**Elliptic and Elliptic++.** Weber et al. (2019) introduced Elliptic and found Random
Forest (illicit-F1 0.796) to outperform GCN (0.628) despite the latter's access to graph
structure, and reported that all models degrade sharply after time step 43, when a dark
market shut down. Elliptic++ (Elmougy and Liu, KDD 2023) adds an address/actor task and,
again, finds Random Forest with feature refinement to be the strongest model.

**Protocol dependence.** Numbers reported on Elliptic vary widely with the evaluation
protocol. Works using random or stratified splits (and weighted F1) report ~0.90-0.98;
works using temporal splits and illicit-class F1 report ~0.6-0.82. These are not
comparable. A verified literature review (this repo, `docs/sota-review.md`) confirms that
under the honest temporal protocol, tree ensembles remain ahead of GNNs.

**Graph anomaly detection and USAD.** Unsupervised graph anomaly detection has a rich
literature - reconstruction autoencoders (DOMINANT, AnomalyDAE, GAD-NR), generative
methods (GAAN, AEGIS), and adversarially-regularized VGAEs (ARGA/ARVGA). USAD (Audibert
et al., KDD 2020), a two-decoder adversarially-trained autoencoder, was designed for
multivariate time series; to our knowledge its two-decoder adversarial reconstruction game
has not been adapted to graphs.

## 3. Protocol

All experiments use a single codebase with fixed seeds and a CI-gated environment. The
honest protocol is: a temporal split (Elliptic's built-in early-vs-late time steps; for
Elliptic++ addresses, a first-seen split), illicit-class PR-AUC and F1 as the primary
metrics (never accuracy or weighted F1), a validation slice carved from the latest train
steps, and a decision threshold chosen on validation and applied to test. The leakage
comparison holds the model, features, and trainer fixed and changes only the split.

## 4. Results

### 4.1 The honest state of the art reproduces "trees beat GNNs"

Under the temporal split and illicit-class F1, XGBoost (PR-AUC 0.799) beats GraphSAGE
(0.488), GAT (0.332) and GCN (0.294), reproducing the 2019 ordering. A heterogeneous GNN
on Elliptic++ improves the best GNN to 0.586 (still below XGBoost). No GNN we trained beats
the tabular baseline under honest evaluation.

### 4.2 The leakage inflation (the headline)

The same models under a random split (Elliptic transactions):

| Model | Temporal PR-AUC | Random PR-AUC | Inflation | Random F1 |
|---|---|---|---|---|
| GCN | 0.294 | 0.800 | +0.506 | 0.736 |
| GraphSAGE | 0.488 | 0.925 | +0.437 | 0.865 |
| GAT | 0.332 | 0.811 | +0.479 | 0.757 |
| XGBoost | 0.790 | **0.987** | +0.197 | **0.955** |

Under the random split every model reaches "SOTA-looking" numbers; XGBoost's 0.987 / 0.955
matches the reported ~0.98. GNNs inflate more than XGBoost, consistent with a double leak:
the random split leaks labels *and* lets message passing cross the train/test boundary. The
effect generalizes to the Elliptic++ **address** task (heterogeneous graph, 822k nodes):
PR-AUC 0.456 -> 0.974, F1 0.529 -> 0.925 (+0.518). It is also robust across seeds: for
GraphSAGE over three seeds the inflation is **+0.427 +/- 0.043 PR-AUC** (temporal 0.495 +/-
0.036 vs random 0.922 +/- 0.008). The inflation is therefore systematic across models,
tasks, and seeds - not an artifact of any single configuration.

**A cross-domain negative control isolates the cause.** On DGraph-Fin (3.7M nodes, a
Chinese fintech social graph with a temporally *stable* fraud rate), the same GraphSAGE
shows essentially **no** inflation: temporal PR-AUC 0.037 vs random 0.039 (+0.002). Where
the distribution does not shift over time, a random split and a temporal split agree, so
there is nothing to leak. This confirms the inflation is a symptom of temporal
distribution shift rather than of random splitting itself - and that Elliptic/Elliptic++,
which shift strongly, are exactly the cases where random-split evaluation is dangerous.
(On DGraph both models are weak in absolute terms, PR-AUC ~0.037 vs a 1.3% base rate;
we report ROC-AUC ~0.74-0.77 for transparency.)

### 4.3 The real open problem: the post-shift collapse

Under the temporal protocol, a rolling per-time-step backtest of a temporal GNN
(EvolveGCN-O) shows PR-AUC collapsing at the shutdown window (steps 44-46: ~0.01) and only
partially recovering. Aggregate numbers hide this; the per-window curve is the honest view.
No surveyed method closes this gap.

### 4.4 A novel unsupervised attempt fails (honest negative)

We implemented, to our knowledge for the first time, a USAD-style two-decoder adversarial
graph autoencoder (GraphUSAD) and two drift-aware variants (rolling-normal window;
domain-adversarial time-invariant representation via gradient reversal). All three score
PR-AUC ~0.037 (below the base rate), matching a plain tabular autoencoder: under this
regime-change shift, reconstruction error tracks the licit distribution's drift rather than
illicitness, regardless of the normal window or latent invariance. We also tried a
*supervised* drift-robust variant - a domain-adversarial GraphSAGE (DANN) that enforces a
time-invariant representation for the supervised classifier - which scored PR-AUC 0.364,
*below* the plain GraphSAGE baseline (0.488): the invariance constraint removed useful
signal rather than improving generalization. Across four principled attempts, enforcing
time-invariance does not solve the shift. The novel method does not work; we report this
honestly, as it maps out what does not, and leaves the post-shift problem open.

## 5. Discussion and recommendations

- **Report illicit-class F1 / PR-AUC under a strict temporal split.** Random-split and
  weighted-F1 numbers are not comparable and overstate progress by up to +0.52 PR-AUC.
- **The post-shift window is the real benchmark.** Progress should be measured on the
  post-time-step-43 regime, where all methods currently fail.
- **Graph structure is not a free win here.** The features already encode aggregated
  neighborhood statistics; under honest evaluation trees remain the model to beat.

## 6. Conclusion

The reported near-perfect performance on Elliptic/Elliptic++ is largely a leakage mirage.
Under leakage-free evaluation the honest state of the art is modest, tree ensembles still
lead, and the temporal distribution shift is unsolved. We provide a reproducible,
CI-gated benchmark and an honest negative result for a novel unsupervised method, and we
argue the field should re-anchor its evaluation.

## Reproducibility

Every number is produced by a CLI command in this repository (`gnn-fraud baselines`,
`train-gnn`, `train-hetero`, `leakage-multi`, `leakage-hetero`, `backtest-temporal`,
`train-graph-usad[-dann]`), under fixed seeds and a locked environment, with a green-gate
and CI. Results JSON in `docs/results/`; the verified literature review in
`docs/sota-review.md`.

## References (anchors)

Peer-reviewed: Weber et al., "Anti-Money Laundering in Bitcoin..." (2019, arXiv 1908.02591);
Elmougy and Liu, Elliptic++ (KDD 2023, arXiv 2306.06108); Audibert et al., USAD (KDD 2020).
Recent preprints (treated as unreplicated) and the full source list are in
`docs/sota-review.md`.
