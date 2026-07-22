# Interview quiz (accumulated)

Self-test questions per step, with answer keys. Written to rehearse for a research
jury. Cover the answer, then check. New steps append here.

---

## Step 0 - Engineering scaffold

**Q1. Why build the green-gate before any ML?**
> A number is only credible if it is reproducible, and an agent is only
> trustworthy if it can verify its own work and stop on red. The gate (lint +
> typecheck + tests + smoke), mirrored by CI, is that trust boundary. *Fix the
> code, never the test.*

**Q2. Why keep torch/PyG as an optional extra rather than a core dependency?**
> To keep the everyday feedback loop fast (seconds). The heavy stack is exercised
> in a separate `--full` / CI job. torch does not need a GPU; the reason is
> install weight, not hardware.

---

## Step 1 - Graph EDA on Elliptic

**Q1. When a GNN does message passing on a node at time step t, where can the
information reaching it come from, and why is there no leakage across the split?**
> Elliptic's edges do not cross time steps (49 connected components ~ 49 time
> steps). So a node at t aggregates only neighbors also at t. There is no edge
> between a train node (t<=34) and a test node (t>=35), so information cannot flow
> through edges across the temporal split. (A "temporal GNN" must therefore model
> time some other way than through inter-time edges.)

**Q2. There are 157,205 "unknown" nodes (77%). What do we do with them?**
> Three roles: (a) train supervised only on the 46,564 labeled; (b) optionally
> pseudo-label high-confidence ones (semi-supervised) - advanced; (c) crucially,
> even when excluded from the loss they *stay in the graph* and carry information
> to labeled nodes during message passing (transductive learning).

**Q3. Why "PR-AUC, not ROC-AUC" under heavy imbalance?**
> ROC's FPR = FP/(FP+TN) has a huge TN denominator, so many false positives
> barely move it -> ROC looks rosy. PR's precision = TP/(TP+FP) compares false
> positives against the rare true positives, so it reflects the real operational
> cost of false alarms.

---

## Step 2 - Non-graph baselines

**Q1. XGBoost scored PR-AUC 0.80 but ROC-AUC 0.93. Why the gap, and which do you
cite?**
> Same imbalance effect as above: ROC-AUC flatters because negatives dominate the
> FPR denominator. Cite PR-AUC (0.80) as the honest headline; mention ROC only to
> show the contrast.

**Q2. The autoencoder scored ROC-AUC 0.21 (below random). Why would inverting the
scores to "get" 0.79 be cheating?**
> Choosing the score orientation using the test labels is fitting to the test set
> (hindsight). The principled model is "anomaly = high reconstruction error"; that
> it fails is the honest result. Inverting would be selecting a hyperparameter on
> test.

**Q3. All three baselines ignore one thing. What, and why might it recover illicit
nodes XGBoost misses (recall only 0.70)?**
> The graph structure (edges / neighborhood). An illicit node can look normal on
> its own features but be betrayed by who it transacts with (mixers, peeling
> chains). Message passing lets a node's neighborhood inform its prediction -
> signal the tabular models cannot see.

---

<!-- Step 3+ quizzes appended here. -->
