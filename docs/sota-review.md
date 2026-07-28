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

**Tree ensembles beat GNNs, unbroken from 2019 to 2026** (by 10+ F1 points in the
literature). GNNs act mainly as representation learners; tabular classifiers give
better decision boundaries under severe imbalance. **Our results reproduce the
trees-over-GNNs ordering** (XGBoost PR-AUC 0.80 / F1 0.82 vs GraphSAGE PR-AUC 0.49 /
F1 0.42). Two honest caveats: our GNN F1 sits well below the literature's honest GNN
values (~0.69 for GraphSAGE) even after a 12-config tuning sweep (best 0.486), so our
tree-vs-GNN gap is larger than the field's; and the within-GNN ordering is
metric-dependent (on F1, GCN edges out SAGE). We lean only on the coarse
trees-over-GNNs conclusion, which is robust.

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

## 5. Novelty of "USAD-on-graph" (NOVEL - dedicated search done)

A dedicated citation / graph-anomaly search (a second verified review) concludes:
**adapting USAD's specific two-decoder adversarial min-max reconstruction game to a
GNN encoder is novel.** USAD appears in graph papers only as a *time-series* baseline;
no published method uses its twin-decoder (AE1/AE2) adversarial reconstruction on a
shared graph encoder. The field splits into separate lineages, none USAD-style:

- **Reconstruction GAEs** (non-adversarial): DOMINANT, AnomalyDAE, GAD-NR, GRASPED, ADA-GAD.
- **Generative GAN-style** (generator/discriminator on representations): GAAN, AEGIS.
- **One-class / pseudo-anomaly**: GGAD (NeurIPS'24), TAM, OCGNN.
- **Adversarially-regularized VGAEs** (latent-space discriminator): ARGA/ARVGA, and the
  **single closest work** - *"Adversarial variational graph autoencoder with contrastive
  learning"* (Journal of Big Data, 2025, CC-BY 4.0): same design space but adversarial in
  the **latent embedding** distribution + DOMINANT-style dual reconstruction, **not**
  USAD's two-decoder reconstruction game.

Positioning bonus: USAD is explicitly motivated by **training stability over GANs**, so a
USAD-on-graph is differentiated from GAAN/AEGIS on that axis too. (Caveat: a brand-new
un-indexed 2025-26 preprint could exist; the claim is scoped to the surveyed literature.)

## 6. Complementary datasets for a temporal-shift multi-dataset study

| Dataset | Nodes / edges | Type | Labels (base rate) | Temporal? | Access |
|---|---|---|---|---|---|
| **DGraph-Fin** | 3.70M / 4.30M | homogeneous | 15,509 fraud / 1.21M normal (~1.3%) | **Yes** (timestamped edges) | NeurIPS'22 track |
| **IBM AMLworld** | synthetic tx graphs | tx graph | full ground truth (AML) | **Yes** (timestamped) | Kaggle (synthetic) |
| Ethereum MulDiGraph (XBlock) | 2.97M / 13.55M | directed multigraph | 1,165 phishing (~0.04%) | coarse | XBlock |
| YelpChi / Amazon | ~45K / ~11K | heterogeneous | ~14.5% / ~9.5% | **No** (static) | CARE-GNN/PC-GNN |

**Recommendation: DGraph-Fin and IBM AMLworld** best complement Elliptic (both permit a
genuine temporal split); YelpChi/Amazon are static; MulDiGraph's temporal signal is coarse.

## 7. Still-hypothesis (flagged honestly)

- Whether "graph structure is a liability under shift" replicates - specific magnitudes
  were refuted in verification; treat as a hypothesis, not a result.

## Sources

Peer-reviewed anchors: Weber et al. 2019 (arXiv 1908.02591); Elliptic++ KDD'23
(arXiv 2306.06108; github.com/git-disl/EllipticPlusPlus); Elliptic Medium post.
Recent (2026) preprints, non-peer-reviewed, some single-author (treat magnitudes as
unreplicated): arXiv 2604.19514, 2603.13998, 2604.23494, 2507.01980, 2511.00047.
Benchmark/context: NeurIPS 2022 datasets track; DGraph; graph-anomaly surveys
(arXiv 2106.07178, 1902.06924).

*Caveat: recorded honestly - some figures come from recent unreplicated preprints;
the peer-reviewed Weber 2019 and Elliptic++ KDD'23 anchors are the most solid.*
