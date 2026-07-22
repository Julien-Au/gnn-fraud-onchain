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

<!-- Step 2+ notes appended here: baselines (LogReg/XGBoost + autoencoder),
message passing, GCN/SAGE/GAT, over-smoothing, heterogeneous/temporal graphs. -->
