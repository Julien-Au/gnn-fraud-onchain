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

## Step 3b - A temporal GNN (EvolveGCN-O)

**Concept.** In Elliptic a transaction exists at a single time step, so there is
no node trajectory to model with a per-node RNN. EvolveGCN instead evolves the GCN
*weights* across the snapshot sequence: a GRU maps W_{t-1} -> W_t, and the conv
applies A_hat @ (X @ W_t). I implemented EvolveGCN-O (Pareja et al., AAAI 2020)
from scratch - the weight is the GRU hidden state fed with the previous weight.

**Result (strict split, train 1-29 / val 30-34 / test 35-49):** test PR-AUC 0.069
- worse than everything. A diagnostic (loss + val + test PR-AUC across LR and
gradient clipping) confirmed: **train loss decreases (it learns), val ~0.3, but
test collapses to ~0.1** in every config. Not a bug; a real generalization failure
across the post-shutdown distribution shift when extrapolating far in time.

**What I must be able to say:**
- **Why it fails here**: far-horizon temporal extrapolation across a regime change
  (dark-market shutdown). The evolved weights, fit on early steps, do not transfer
  to the very different late steps.
- **Why the comparison is not yet fair**: the static GNNs were transductive
  (test-node features seen during message passing); EvolveGCN got no test-period
  access. And EvolveGCN's canonical protocol is *rolling* (predict step t+1 from
  a model trained up to t), not one far split - the fair experiment is queued.
- **BPTT stability**: I checked gradient clipping and lower LR; they change the
  trajectory slightly but not the conclusion. Good habit to verify before blaming
  the architecture.
- **The meta-lesson**: a suspicious number (0.069, below the base rate) gets
  *diagnosed*, not reported blindly. The diagnostic is what turns "the model is
  bad" into "the model learns but cannot extrapolate across the shift" - a claim I
  can defend.

## Step 4 - Heterogeneous graph (Elliptic++): the graph earns its keep

**Concept.** A heterogeneous graph has multiple node and edge types. Elliptic++ adds
`addr` (wallet) nodes to the `tx` nodes, with relations tx-tx, addr-tx, tx-addr,
addr-addr. A `HeteroConv` GNN keeps one message-passing function per relation and
aggregates per destination node type - so the model is parameterized by the schema,
which is the bridge to the "one model over many relational schemas" idea.

**Result (same tx labels/split as before, so comparable):** heterogeneous SAGE
**PR-AUC 0.586**, up from 0.488 for the tx-only GraphSAGE - **+20% relative, the
biggest graph-driven gain in the project.** Still below XGBoost (0.80), but the
trend is the point.

**What I must be able to say (this is the money slide):**
- **Why the hetero graph helps**: illicit transactions are betrayed by the
  *addresses* they touch (shared wallets, address-level flow) - signal that lives
  in the addr-tx and addr-addr relations, invisible to a tx-only graph and to the
  tabular features. Adding that structure is what closed part of the gap to XGBoost.
- **The relational-foundation-model link**: the same encoder definition adapted to
  a new schema (two node types, four relations) with no architecture rewrite -
  HeteroConv/`to_hetero` parameterize by `metadata`. Scale that idea across many
  schemas and you get a relational foundation model.
- **Honest caveats**: val PR-AUC 0.96 >> test 0.59 - the temporal shift still hurts
  generalization; address features are mean-aggregated per address (addresses span
  many time steps, edges are timestamp-free), a documented modeling choice.
- **Engineering reality**: 2 GB of CSVs -> a 1M-node / 8.8M-edge HeteroData; hit
  and fixed a NaN-propagation bug (unnormalized wallet features with huge scales)
  by cleaning + z-scoring features; ran training on a throwaway cloud box (local
  was too slow) and destroyed it.

## Step 5 extras - rolling backtest, address task, Docker

**Rolling temporal backtest (EvolveGCN's fair shot).** With a fuller training budget
and a *per-time-step* rolling evaluation, aggregate test PR-AUC rises only to 0.100.
The value is the breakdown: PR-AUC collapses at the dark-market shutdown (steps
44-46: 0.012 / 0.010 / 0.005) and recovers after. Lesson to say out loud: *when a
temporal model fails, show the failure over the horizon* - an aggregate number hides
a regime change that a per-step curve makes obvious.

**Address-level classification.** Pointing the same HeteroConv model at the `addr`
node type instead of `tx` detects illicit wallets: PR-AUC 0.456, F1 0.529 on 92,451
test addresses. The engineering point is the payoff of a *schema-parameterized*
model: `train-hetero --target addr` reuses the exact architecture with a one-word
change. That is the concrete version of "one model over many relational schemas."
Modeling choice: addresses split by **first-seen** time step (they span many steps).

**Docker.** A `uv`-based image (`docker build ...`) gives a second, OS-level
reproducibility guarantee on top of the locked env, and a CI `docker` job smoke-runs
the CLI in the image. The core image stays small; the graph stack is an opt-in build
arg (`--build-arg EXTRAS="--extra gnn"`).
