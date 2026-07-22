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

<!-- Step 1+ notes appended here: graph EDA, message passing, GCN/SAGE/GAT,
over-smoothing, heterogeneous graphs, temporal GNNs, imbalanced metrics. -->
