# SOTA review: fraud detection on Elliptic / Elliptic++

Findings from a multi-source, adversarially-verified literature review (25 claims
vetted, 22 confirmed / 3 refuted). Sources listed at the end. Peer-review status is
flagged where it matters. **This review's headline is that the Elliptic "SOTA" is
protocol-dependent, and only temporally-split, illicit-class numbers are comparable.**

## 1. Comparable SOTA (honest temporal protocol)

Under a **strict temporal split** (train time steps 1-34, test 35-49) and the
**illicit-class (minority) F1** metric - the only comparable, honest setting:

| Model | Illicit-class F1 (temporal) | Source |
|---|---|---|
| Random Forest (raw features) | ~0.79-0.82 | Weber 2019; 2026 re-eval |
| RF + GCN embeddings | 0.796 | Weber et al. 2019 |
| GraphSAGE | ~0.69 | 2026 re-eval |
| Skip-GCN | 0.705 | Weber et al. 2019 |
| GAT / GCN | ~0.61 / 0.50-0.63 | Weber 2019 / 2026 re-eval |

**Tree ensembles beat GNNs by 10+ points, unbroken from 2019 to 2026.** GNNs act
mainly as representation learners; tabular classifiers give better decision
boundaries under severe imbalance. **Our own results reproduce this ordering exactly**
(XGBoost 0.80 F1 >> GraphSAGE 0.49 >> GAT/GCN), an independent honest confirmation.

## 2. The leakage warning (why "~0.98 SOTA" is not real)

Papers reporting ~0.90-0.98 on Elliptic/Elliptic++ (e.g. ChronoWave-GNN 0.98,
SAGE-FIN) use **random / stratified / transductive splits that leak future time
steps**, and often report **overall/weighted** F1 (majority-dominated), not
illicit-class F1. These numbers **must not be ranked against temporal-protocol
results**. Even under a random split on Elliptic++, XGBoost (0.89 F1) beats the GNN.
Chasing a leaked 0.98 would be scientifically dishonest - a reviewer rejects it first.

## 3. The open problem: the post-time-step-43 collapse (unsolved)

At time step 43 (a dark-market shutdown) the fraud base rate collapses ~39x
(~11.6% -> ~0.3%); per-window illicit-F1 falls from ~0.38 (steps 35-42) to ~0.03
(steps 43-49). **No surveyed method closes this gap under a strict temporal
protocol** - drift handling is repeatedly listed only as future work. **Our rolling
backtest independently reproduced this collapse** (per-step PR-AUC ~0.01 at t44-46).

## 4. Elliptic++ (KDD'23)

Adds an 822,942-address actor task + augmented features. Its own best model is again
**Random Forest with feature refinement** (transactions 98.6%P / 72.7%R; actors
92.1%P / 80.2%R, under a temporal 70/30 split), beating MLP/XGBoost/LSTM/LR.

## 5. Novelty of "USAD-on-graph" (under-explored, not proven absent)

The surveyed graph-fraud literature makes **no use of USAD or its two-decoder
adversarial autoencoder**; dynamic/temporal graph anomaly detection is listed as
future work. This supports "USAD-on-graph targeting temporal drift" as a **genuine,
under-explored gap** - but the verified claim set did **not** include a dedicated
citation-graph search of USAD-citing works or a systematic sweep of adversarial
graph autoencoders (DOMINANT, AnomalyDAE, CoLA, adversarial GAE/VGAE). **Before any
novelty claim in a paper, that dedicated search is required.**

## 6. Still open (not answered by the verified set)

- Complementary benchmark datasets (Ethereum phishing, AMLSim/AMLworld, DGraph-Fin,
  YelpChi/Amazon) with counts/labels/licenses - unanswered; needs a targeted pass.
- A conclusive USAD-novelty citation search (Semantic Scholar / citing-works).
- Whether the "graph structure is a liability under shift" finding replicates (its
  specific magnitudes were refuted in verification; treat as a hypothesis).

## Sources

Peer-reviewed anchors: Weber et al. 2019 (arXiv 1908.02591); Elliptic++ KDD'23
(arXiv 2306.06108; github.com/git-disl/EllipticPlusPlus); Elliptic Medium post.
Recent (2026) preprints, non-peer-reviewed, some single-author (treat magnitudes as
unreplicated): arXiv 2604.19514, 2603.13998, 2604.23494, 2507.01980, 2511.00047.
Benchmark/context: NeurIPS 2022 datasets track; DGraph; graph-anomaly surveys
(arXiv 2106.07178, 1902.06924).

*Caveat: recorded honestly - some figures come from recent unreplicated preprints;
the peer-reviewed Weber 2019 and Elliptic++ KDD'23 anchors are the most solid.*
