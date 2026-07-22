# Learning notes

Interview-ready notes, written as I build. Each entry is something I should be
able to re-explain out loud to a research jury. Kept in English on purpose.

Target framing: connect my prior work (USAD, KDD 2020 - unsupervised anomaly
detection on multivariate time series with adversarially-trained autoencoders)
to **graph / relational learning**, which is the gap this project closes.

---

## Step 0 - Why the engineering scaffold comes first

**Concept.** Before any model, I stand up a *self-verifying loop*: a green-gate
(`scripts/verify.sh` = lint + typecheck + tests + smoke) mirrored by CI. The unit
of trust is "the gate is green", not "it ran on my machine".

**Why it matters (and why a jury cares).** Research credibility = reproducibility.
Fixed seeds, a locked environment (`uv`), one config file per experiment, and a
CI that re-runs the checks are what let someone else - or future me - reproduce a
number. The same discipline is what makes an *autonomous agent* trustworthy: it
can only be let loose if it can check its own work and halt on red.

**Trade-off chosen.** The graph stack (torch, torch-geometric) is an *optional
extra*, not a core dep, so the everyday gate runs in seconds without pulling a
big ML stack; a separate CI job exercises the heavy path. Cost: two tiers to
maintain. Benefit: fast feedback, which is the whole point of a loop.

**Things I should be able to say:**
- "PR-AUC and minority-class F1, not accuracy" - and *why* (see step 2 note when
  written): with ~2% positives, a model predicting all-negative scores 98%
  accuracy and is useless; PR-AUC focuses on the positive class.
- "Temporal split, no leakage" - with 49 time steps, train on early steps, test
  on later ones; random splits leak future structure into training.
- The green-gate philosophy: *fix the code, never the test.*

**Open questions to resolve in later steps:** exact class balance and label
coverage of Elliptic (step 1); whether transaction-level or address-level nodes
better match the "relational schema" story (step 4).

---

## Step 1 - Graph EDA on Elliptic (real numbers)

**Concept.** A PyG graph is three things: `x` (node features `[N, F]`),
`edge_index` (edges `[2, E]`), `y` (labels). The edges are the *relational*
signal a tabular model cannot see. EDA's job is to characterize that signal and
the label structure before modeling.

**What I measured (with `gnn-fraud eda`, not recited):**
- 203,769 nodes, 234,355 edges, 165 features, directed.
- Classes: 42,019 licit / 4,545 illicit / 157,205 unknown. Only 46,564 nodes are
  labeled; illicit = **9.76% of labeled, 2.23% of all**. This is the imbalance
  that makes accuracy meaningless (predicting "all licit" scores ~98%).
- Degree: mean 2.30, max 473, **0 isolated** -> a sparse, heavy-tailed graph.
- **49 connected components for 49 time steps.** Big structural insight: edges
  almost never cross time steps, so the graph is nearly a disjoint union of
  per-time-step subgraphs. Consequence: a node's receptive field under message
  passing is confined to its own time step - there is no leakage *through edges*
  across the temporal split, only through shared model weights.
- Built-in temporal split keeps positives on both sides (3,462 train / 1,083
  test illicit), so PR-AUC/F1 are estimable on test.

**Things I should be able to say out loud:**
- Why 165 vs 166 features (PyG build detail; original = 94 local + 72 aggregated
  + time step).
- Why the temporal split is *already* honest here, and why 49 components ~ 49
  time steps is not a coincidence.
- "PR-AUC + minority F1 + confusion matrix", and why ROC-AUC would flatter us.

**Trade-off in the EDA code.** I first looped PyG's `EllipticBitcoinTemporalDataset`
over 49 time steps for the temporal figure - correct but very slow (it reprocesses
each call). I switched to reading the raw CSV once with pandas and grouping by
the time-step column: same numbers, deterministic, seconds not minutes. Lesson:
prefer the cheapest source of truth for repeated reads.

## Step 2 - Non-graph baselines (the honest reference)

**Concept.** Before any GNN, fit models that see only the 165 node features and
*not* the graph. If a GNN later beats these, the gain is attributable to the
graph. Three baselines, three philosophies: logistic regression (linear floor),
XGBoost (the strong tabular bar), and an unsupervised autoencoder (the USAD
instinct: train on normal, score by reconstruction error).

**Protocol (defensible).** Scaler fit on **train only** (no leakage); the built-in
**temporal** split; F1 threshold chosen on **train** then applied to test; PR-AUC
primary.

**Real results (temporal test):**
| model | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| logreg | 0.288 | 0.881 | 0.351 |
| xgboost | **0.799** | 0.928 | **0.817** |
| autoencoder | 0.038 | 0.213 | 0.122 |

**What I must be able to explain:**
- **PR-AUC vs ROC-AUC, live:** XGBoost's ROC-AUC (0.93) flatters vs its PR-AUC
  (0.80). ROC's FPR has a giant TN denominator, so false positives barely move it;
  PR-AUC's precision compares FP against the rare TP, so it "feels" them. In heavy
  imbalance, report PR-AUC.
- **Why the autoencoder is worse than random (ROC 0.21).** It learns "normal" from
  early-time-step licit nodes, but the licit distribution *shifts* after the
  dark-market shutdown. At test time, test-period licit reconstruct worse than
  illicit, so high reconstruction error no longer means "illicit". This is a
  genuine failure of the stationarity assumption behind reconstruction-based
  anomaly detection - and a clean motivation for structure (graph) and explicit
  temporal handling. I do **not** invert the score using test knowledge; the
  principled "anomaly = high error" model is what is reported.
- **The USAD link, precisely.** USAD worked because (a) time series are (locally)
  stationary and (b) an adversarial second decoder sharpened the boundary. Here
  (a) is violated by the temporal shift, and I used only the plain AE (no
  adversarial part). So the baseline is honest about what carries over and what
  does not.

## Step 3 - First GNNs (GCN -> GraphSAGE -> GAT)

**Concept: message passing.** Each layer updates a node by aggregating its
neighbors' representations, then transforming. Stack L layers -> a node sees its
L-hop neighborhood. The three architectures differ in *how* they aggregate:
- **GCN**: normalized mean of neighbors (symmetric degree normalization), one
  shared linear map. Simple, no neighbor weighting.
- **GraphSAGE**: aggregate neighbors, then combine with the node's *own* previous
  representation (self vs neighbors kept distinct); designed to be inductive.
- **GAT**: learn attention weights so some neighbors count more; multi-head.

**Real results (transductive, temporal test):** GCN 0.294 / SAGE **0.488** / GAT
0.332 PR-AUC, vs XGBoost **0.799**. The graph does **not** beat the tabular
baseline yet.

**What I must be able to explain (this is the interesting part):**
- **Why GNNs lose to XGBoost on Elliptic.** (1) The features already contain 72
  *aggregated neighbor* features, so much graph signal is in XGBoost's input too.
  (2) The graph is disconnected per time step, so message passing has a small
  receptive field. (3) Temporal shift: a transductive GNN has no mechanism to
  adapt from early to late steps. This is a documented result (Weber et al. 2019).
- **Why SAGE > GCN > GAT here.** Keeping self distinct from neighbor aggregation
  (SAGE) helps; attention (GAT) does not, plausibly because mean degree is only
  2.3 - with ~2 neighbors there is little for attention to weight, and the extra
  parameters just add variance.
- **Over-smoothing** (why 2 layers, not 10): stacking many message-passing layers
  makes all node representations converge to similar values (repeated averaging ->
  a low-pass filter), erasing the distinctions we need. Shallow GNNs sidestep it.
- **Transductive vs inductive**: here we train and test on one fixed graph
  (transductive) and let unknown nodes carry messages; SAGE would also allow
  scoring brand-new nodes (inductive), relevant for deployment.
- **Honesty framing for the jury**: a negative result, correctly analyzed, is
  worth more than an inflated one. It sets up step 4 (temporal GNN + heterogeneous
  graph) as a *motivated* next step, not a reflex.

**Compute note.** Ran the training on a throwaway Hetzner box (the local machine
was busy), then destroyed it. Repro is unchanged: same code, same seeds, CPU.

<!-- Step 4+ notes: temporal GNN (EvolveGCN), heterogeneous graph (Elliptic++),
relational foundation model framing. -->
