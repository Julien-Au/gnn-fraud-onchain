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

## Step 3 - First GNNs

**Q1. In one sentence each, what does GCN, GraphSAGE, and GAT change relative to
the previous one?**
> GCN: normalized mean of neighbors. GraphSAGE: keep the node's own representation
> distinct from the aggregated neighbors (and aim for inductive use). GAT: learn
> attention weights so neighbors are not weighted equally.

**Q2. On Elliptic the best GNN (SAGE, PR-AUC 0.49) loses to XGBoost (0.80). Give
three reasons.**
> (1) The 165 features already include 72 aggregated-neighbor features, so graph
> signal is partly in XGBoost's input. (2) The graph is disconnected per time step,
> so message passing has a small receptive field. (3) Temporal distribution shift
> (dark-market shutdown) - a transductive GNN has no mechanism to adapt over time.

**Q3. Why does attention (GAT) not help here, and what is over-smoothing?**
> Mean degree is ~2.3, so with about two neighbors there is little for attention
> to weight; the extra parameters mostly add variance. Over-smoothing: stacking
> many message-passing layers repeatedly averages neighborhoods, driving all node
> representations toward the same value and erasing useful distinctions - which is
> why we use shallow (2-layer) GNNs.

**Q4. Why is "the graph does not help yet" a good result to present, not a failure
to hide?**
> Because it is real and it is *explained*, and it motivates the next step
> (temporal + heterogeneous modeling) on evidence rather than reflex. A jury
> trusts an analyzed negative result more than an unexplained win.

---

## Step 3b - Temporal GNN (EvolveGCN-O)

**Q1. In Elliptic a transaction exists at one time step. Why does that rule out a
per-node RNN, and what does EvolveGCN evolve instead?**
> There is no node trajectory to feed an RNN (a transaction is not "the same node"
> at t and t+1). EvolveGCN evolves the GCN *weight matrices* across snapshots with
> a GRU (W_{t-1} -> W_t), so the model's parameters adapt over time.

**Q2. EvolveGCN scored test PR-AUC 0.069. How did you show this is a real result,
not a bug?**
> A committed lr x gradient-clipping sweep (docs/results/evolvegcn_sweep.json): the
> training loss decreases steadily (it learns), val PR-AUC reaches up to 0.459, but
> test PR-AUC stays at 0.10-0.20 in every configuration (best 0.200) - a
> generalization failure across the post-shutdown shift, not a training bug.

**Q3. Why is comparing EvolveGCN's 0.069 to the static GNNs' ~0.4 not yet fair?**
> The static GNNs were transductive (test-node features participated in message
> passing); EvolveGCN had no test-period access and extrapolated far in time.
> EvolveGCN's canonical protocol is also rolling (predict t+1 from a model trained
> up to t), not one far split - the fair experiment is still to run.

---

## Step 4 - Heterogeneous GNN (Elliptic++)

**Q1. What makes Elliptic++ a heterogeneous graph, and how does a HeteroConv GNN
handle it?**
> It has two node types (transactions, addresses) and four relations (tx-tx,
> addr-tx, tx-addr, addr-addr). HeteroConv keeps one convolution per relation and
> aggregates the incoming messages per destination node type, so the model is
> parameterized by the schema rather than hard-coded to one graph.

**Q2. The hetero GNN scored 0.586 vs 0.488 for the tx-only GNN. Why does the
address structure help, and why is the comparison fair?**
> Illicit transactions are betrayed by the addresses they touch (shared wallets,
> address-level flow) - signal in the addr-tx / addr-addr relations that a tx-only
> graph and the tabular features miss. It is fair because the task, labels and
> temporal split are identical; only the graph changed.

**Q3. This is the "relational foundation model" pitch in miniature. Explain the
link.**
> The same encoder definition adapted to a new schema (2 node types, 4 relations)
> with no architecture rewrite - it is parameterized by the graph's metadata.
> Generalize that across many schemas and you get one model over many relational
> structures. And empirically, a richer relational schema gave a better model.

**Q4. Name two honest caveats you would raise yourself.**
> Validation PR-AUC (0.96) is far above test (0.59), so the temporal shift still
> hurts generalization; and address features are mean-aggregated per address (a
> modeling choice, since addresses span many time steps and edges are timeless).

---

## Step 5 extras

**Q1. EvolveGCN's aggregate PR-AUC only rose to 0.100 with a fair rolling protocol.
Why is the per-time-step backtest more informative than that number?**
> Because the aggregate hides *where* it fails. The per-step curve shows PR-AUC
> collapsing at the dark-market shutdown (steps 44-46 ~0.01) and recovering after -
> pinpointing the regime change as the cause. Always show a temporal model's failure
> over the horizon.

**Q2. You classified addresses with the same model that classified transactions.
What did that require, and why does it matter?**
> Only a one-word change (`--target addr`): the HeteroConv model is parameterized by
> the schema, so it serves either node type with no architecture change. It matters
> because it is the concrete, working version of "one model over many relational
> schemas" - the relational-foundation-model thesis.

**Q3. Why split addresses by "first-seen" time step, and why is the address result
not directly comparable to the transaction result?**
> Addresses span many time steps (unlike transactions), so there is no single
> timestamp; first-seen gives a defensible temporal split. It is a different task
> with a different label set and test set, so the PR-AUC is not comparable to the tx
> numbers - the value is the shared pipeline, not a head-to-head score.

---

<!-- Later quizzes appended here. -->




