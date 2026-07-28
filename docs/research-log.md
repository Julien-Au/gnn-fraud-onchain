# Research log

Post-benchmark research phase: experiments toward a genuine contribution, logged
honestly - negative results included, because ruling out a hypothesis is progress.
All numbers are real and reproducible.

Context: on Elliptic, gradient-boosted trees (XGBoost, PR-AUC 0.799) beat every GNN
we tried; the heterogeneous graph helps (0.586) but does not close the gap; and the
open problem is the temporal distribution shift (a dark-market shutdown around time
step 43 collapses every method's performance - see the rolling backtest figure).

---

## Exp 1 - GraphUSAD v1 (naive USAD-on-graph): NEGATIVE

**Hypothesis.** Porting USAD (adversarial two-decoder autoencoder, KDD 2020) to a
graph - GNN encoder + two feature-reconstruction decoders, trained on licit nodes,
anomaly = reconstruction error - would beat the plain tabular autoencoder by using
graph context and the adversarial boundary-sharpening.

**Setup.** `models/graph_usad.py` + `train/graph_usad_trainer.py`
(`gnn-fraud train-graph-usad`). GraphSAGE encoder, MLP decoders, USAD 1/n adversarial
schedule, GAN-style two-optimizer split. Trained on licit train nodes; evaluated on
the temporal test split; seed 42.

**Result.** PR-AUC **0.037**, ROC-AUC **0.179** (below random) - essentially
identical to the plain tabular autoencoder (PR-AUC 0.038, ROC-AUC 0.213). Same
failure mode: illicit test nodes reconstruct *better* than test-period licit nodes,
so the threshold collapses to predict-all-positive.

**Conclusion.** Graph-ifying USAD does **not** help as-is. The bottleneck is not the
model class but the **distribution shift**: any reconstruction-based unsupervised
detector fails when "normal" drifts between train and test periods. This rules out
the naive approach and sharpens the direction:

**Next hypotheses (to prioritize after the literature review):**
- Train the "normal" model on a **rolling / most-recent window** rather than all
  early steps, so it tracks the drifting normal (drift-aware unsupervised detection).
- **Test-time adaptation** of the encoder to the test period's distribution.
- A **discriminative** shift-robust objective instead of reconstruction.
- Positioning vs the literature (is USAD-on-graph already published? what do
  shift-aware graph fraud methods report?) - pending the deep-research report.

---

## Exp 2 - Leakage demonstration (temporal vs random split): POSITIVE

**Hypothesis.** The literature's near-perfect Elliptic numbers (~0.85-0.98) come
mostly from evaluation leakage (random splits that leak future time steps), not from
better models. If so, the *same* model should look near-SOTA under a random split
and mediocre under the honest temporal split.

**Setup.** `experiments/leakage.py` (`gnn-fraud leakage --model sage`). The exact
same GraphSAGE, trained/evaluated under (a) the honest temporal split (train steps
1-34 / test 35-49) and (b) a stratified random split of the labeled nodes. Seed 42.

**Result (confirmed):**
| Split | PR-AUC | F1 (illicit) | ROC-AUC |
|---|---|---|---|
| Temporal (honest) | 0.488 | 0.429 | 0.877 |
| **Random (leaky)** | **0.925** | **0.857** | 0.983 |

**+0.437 PR-AUC and F1 0.43 -> 0.86 from the split alone.** Under the random split
our vanilla GraphSAGE lands at F1 0.857 - squarely inside the "SOTA" range reported
by papers that use random/stratified splits - while the identical model scores 0.43
under honest temporal evaluation. This independently reproduces, on our own code, the
inflation the SOTA review flagged: the high numbers are a protocol artifact.

**Why this matters (Path A contribution).** This is a clean, self-contained,
reproducible demonstration for a leakage-free re-evaluation / benchmark contribution:
"report illicit-class F1 under a strict temporal split, or you are measuring
leakage." It also fixes the honest baseline the drift-robust method (Path B) must
improve: PR-AUC 0.488 (SAGE) / 0.799 (XGBoost) under temporal evaluation.

---

## Exp 3 - GraphUSAD v2 (drift-aware rolling normal): NEGATIVE

**Hypothesis.** v1 failed because reconstruction error tracked the licit
distribution's *drift* rather than illicitness. Training the "normal" on only the
**latest** train time steps (a rolling window closer to the test period) should fix
the anti-correlation.

**Setup.** `train-graph-usad --normal-recent 5` (normal = train steps 25-29 only).

**Result.** PR-AUC **0.0367**, ROC-AUC 0.180 - **identical to v1** (0.0366). No
improvement; same predict-all-positive collapse.

**Conclusion.** The rolling-normal fix does **not** work. The problem is deeper than
which normal window: under this regime-change shift, illicit test nodes reconstruct
*better* than test-period licit nodes regardless of the normal set, so
reconstruction error is the wrong signal. This rules out the simple drift-aware fix
and, together with Exp 1, casts doubt on the whole reconstruction-based frame for
this shift.

**Honest reassessment.** Two negatives on the unsupervised USAD-on-graph angle. It is
*novel* but not *working* on Elliptic. The next distinct hypotheses would be (a) a
time-domain-invariant representation (adversarially remove the time-period signal so
reconstruction reflects abnormality, not drift), or (b) abandoning reconstruction for
a drift-robust *supervised* method targeting the post-t43 window (supervised models
already dominate: XGBoost 0.80 vs any AE ~0.04). The solid, low-risk contribution
remains the leakage-free benchmark (Exp 2 + the SOTA review).

---

## Exp 4 - Multi-model leakage benchmark (C1 consolidated): STRONG

**Setup.** `gnn-fraud leakage-multi`: GCN, GraphSAGE, GAT and XGBoost, each under the
honest temporal split vs a stratified random split (same model, only the split differs).

**Result (PR-AUC / illicit-F1 under random split):**
| Model | Temporal PR-AUC | Random PR-AUC | Inflation | Random F1 |
|---|---|---|---|---|
| GCN | 0.294 | 0.800 | +0.506 | 0.736 |
| GraphSAGE | 0.488 | 0.925 | +0.437 | 0.865 |
| GAT | 0.332 | 0.811 | +0.479 | 0.757 |
| XGBoost | 0.790 | **0.987** | +0.197 | **0.955** |

**Every model reaches "SOTA-looking" numbers under the leaky random split** - XGBoost
hits PR-AUC 0.987 / F1 0.955, exactly the reported "~0.98 SOTA", and collapses to 0.79
under honest temporal evaluation. This is a clean, self-contained benchmark result:
the near-perfect Elliptic numbers in the literature are reproducible by ANY model
through leakage. **GNNs inflate more (+0.44-0.51) than XGBoost (+0.20)** - a double
leak: the random split leaks labels *and* lets message passing cross the train/test
boundary. This is the core Path-A (C1) contribution.

## Exp 5 - GraphUSAD v3 (domain-adversarial, time-invariant): NEGATIVE

**Hypothesis.** Make the latent time-INVARIANT via a gradient-reversal domain
classifier (train-period vs later), so reconstruction error reflects abnormality, not
drift - the principled fix for v1/v2's failure.

**Setup.** `train-graph-usad-dann` (DANN-style GRL + time-period head, 1/n... lambda ramp).

**Result.** PR-AUC **0.0367**, ROC-AUC 0.181, best epoch 0 - **identical to v1/v2**.
Time-invariance does not help either.

**Conclusion (definitive for the unsupervised angle).** Three principled variants -
naive (Exp 1), rolling-normal (Exp 3), domain-adversarial (Exp 5) - all fail
identically. The reconstruction-based unsupervised frame is the wrong tool for this
regime-change shift, full stop. USAD-on-graph is *novel* but does not work on Elliptic.
This is an honest negative result. The consolidated, defensible contribution is the
leakage-free benchmark (Exp 2 + Exp 4 + the SOTA review): a reality check showing the
reported Elliptic SOTA is a leakage artifact.

---

## Exp 6 - Leakage generalizes to a second task (Elliptic++ addresses): STRONG

**Hypothesis.** If the inflation is a systematic property of random splits on temporal
graphs (not an Elliptic-transactions quirk), it should reproduce on a structurally
different task.

**Setup.** `gnn-fraud leakage-hetero --target addr`: the heterogeneous GNN on the
Elliptic++ **address** task (822,942 address nodes, first-seen temporal split), under
the honest temporal split vs a stratified random split.

**Result (confirmed):**
| Split | PR-AUC | F1 |
|---|---|---|
| Temporal (honest) | 0.456 | 0.529 |
| **Random (leaky)** | **0.974** | **0.925** |

**+0.518 PR-AUC from the split alone** - PR-AUC 0.46 -> 0.97, F1 0.53 -> 0.93 - on a
heterogeneous graph and a different node type. The inflation is **not** an
Elliptic-transactions artifact; it is systematic (consistent with the +0.44-0.51 on
the transaction task in Exp 4). This is the multi-task/multi-graph generalization that
makes the leakage critique (C1) robust, not anecdotal.

**Status.** C1 (leakage-free benchmark reality-check) now demonstrated on the Elliptic
transaction task (4 models) and the Elliptic++ address task - a solid, generalizable,
publishable contribution. Optional further strengthening: a fully different-domain
dataset (DGraph-Fin, gated behind registration).

---

## Exp 7 - Supervised drift-robust GNN (domain-adversarial DANN): NEGATIVE

**Hypothesis.** Keep the strong supervised signal (unlike the unsupervised AEs) and add
a gradient-reversal time-period discriminator, so the representation is time-invariant
and generalizes better to the post-shift test period. Should beat plain GraphSAGE (0.488).

**Setup.** `gnn-fraud train-dann`: GraphSAGE encoder + supervised classifier + GRL
time-period head; DANN lambda ramp; temporal split; seed 42.

**Result.** PR-AUC **0.364**, F1 0.388 - **below** plain GraphSAGE (0.488). Validation
PR-AUC 0.85 but test 0.36: the domain-adversarial constraint removed useful signal
rather than improving generalization; the shift still dominates.

**Conclusion.** The supervised time-invariance approach also fails to beat the baseline.
Across four method attempts (unsupervised USAD x3, supervised DANN x1), none beats the
honest baselines: enforcing time-invariance does not solve Elliptic's regime-change
shift. This closes the bounded method-exploration with an honest negative.

## Exp 8 - Multi-seed robustness of the leakage inflation (C1 strengthened): STRONG

**Setup.** `gnn-fraud leakage-seeds` (GraphSAGE, seeds 42/43/44): temporal vs random split.

**Result.** temporal PR-AUC 0.495 +/- 0.036; random 0.922 +/- 0.008; **inflation
+0.427 +/- 0.043 PR-AUC (n=3)**. The leakage inflation is robust across seeds, not a
single-seed artifact. With the Elliptic++ address generalization (+0.52), C1 is
statistically and across-task solid.

## Exp 9 - DGraph-Fin cross-domain: a negative control that reveals the mechanism

**Setup.** `gnn-fraud leakage-dgraph`: the same GraphSAGE on DGraph-Fin (3.7M nodes,
4.3M timestamped edges, ~1.3% fraud; a Chinese fintech social graph, gated dataset),
under a temporal (first-seen) split vs a random split.

**Result (informative, not what we expected):**
| Split | PR-AUC | F1 | ROC-AUC |
|---|---|---|---|
| Temporal (honest) | 0.037 | 0.079 | 0.736 |
| Random (leaky) | 0.039 | 0.085 | 0.767 |

**Inflation +0.002 PR-AUC - essentially none**, in sharp contrast to Elliptic
(+0.44-0.52). Two honest reads, both refining the thesis:
1. **Mechanism.** DGraph's fraud rate is roughly temporally stable (1.2% -> 1.4%),
   so a random split and a temporal split are similar - there is no future to leak.
   The Elliptic inflation is therefore driven by its strong temporal distribution
   shift (the dark-market shutdown), not by random splitting per se. DGraph acts as a
   **negative control** that isolates the cause.
2. **Caveat.** On DGraph, GraphSAGE is weak in absolute terms (PR-AUC ~0.037, a few x
   the 1.3% base rate; ROC-AUC ~0.74), so the comparison is between two weak models;
   we report ROC-AUC alongside PR-AUC for transparency.

**Refined claim (stronger and more honest).** Leakage inflation is not universal - it
scales with the temporal distribution shift. Where the shift is strong (Elliptic,
Elliptic++), random-split evaluation is catastrophically misleading (+0.44-0.52
PR-AUC); where the distribution is stable (DGraph), it is negligible. The practical
warning stands and is sharper: *on temporally-shifting fraud graphs, report a temporal
split* - and the popular Elliptic benchmark is exactly such a case.

## Exp 10 - Gap decomposition (5 seeds): our "double leak" hypothesis REFUTED

**Question.** What fraction of the temporal-vs-random gap is actual leakage, versus
prevalence arithmetic, versus "distribution access" (training on test-period data)?

**Setup.** `gnn-fraud decompose`: four arms on threshold-free PR-AUC, 5 seeds -
temporal / random / random with test prevalence matched to the temporal test (6.5%) /
prevalence-matched with **inductive training** (all test nodes removed from the graph,
so no test feature or edge is visible during training).

**Result (GraphSAGE, mean +/- std over 5 seeds):**
| Component of the +0.410 +/- 0.041 gap | Value |
|---|---|
| Base-rate (prevalence arithmetic) | +0.026 +/- 0.005 |
| **Message-passing leakage (our "double leak")** | **-0.000 +/- 0.007 - ZERO** |
| **Distribution access (training on the test period)** | **+0.384 +/- 0.044 - dominant** |

XGBoost (no graph): total +0.195 +/- 0.001 = base-rate +0.005 + distribution access
+0.189.

**Conclusion (honest self-correction).** The inductive ablation (0.894 vs 0.894 -
identical) refutes the message-passing "double leak" we had hypothesized: there is no
graph-specific leakage channel here. GNNs inflate more than XGBoost (+0.41-0.55 vs
+0.20) because they **generalize worse across the temporal shift** - their temporal
arm is weaker - so random-split "distribution access" rescues them more. The
evaluation flaw in the literature is real, but its mechanism is distribution access,
not graph leakage. We proposed the double-leak, measured it, and it is null; the
paper now says so.

## Exp 11 - GNN tuning sweep (12 configs, temporal val selection): modest

**Result.** Best SAGE (hidden 128, lr 0.005, dropout 0.5): test PR-AUC 0.506 / F1
0.486 (vs 0.488 / 0.422 default). Best GCN (256/0.01/0.3): PR-AUC 0.274. Tuning
improves F1 by ~6 points but does **not** close the ~20-point gap to the literature's
honest-protocol GNN F1 (~0.69); the residual likely reflects protocol/training-detail
differences in those (unreplicated) reports. Reported as measured.

## Exp 12 - Deployment-honest hetero arm (pre_split features): conclusion robust

**Setup.** `leakage-hetero --feature-window pre_split`: wallet features aggregated
over pre-split activity only; z-scoring fit on train-period nodes only (fixes the
committee-identified feature leakage in the "honest" arm).

**Result.** Temporal 0.469 / random 0.966 -> inflation **+0.497**, vs +0.518 with
lifetime features (temporal 0.456). Fixing the feature leak barely moves either arm:
the address-task inflation is robust to the honest feature protocol.

## Overall verdict

Two clean contributions and four honest negatives. **C1 (the leakage-free reality
check) is the real, defensible, publishable result**: the reported Elliptic/Elliptic++
SOTA is a leakage artifact (inflation +0.20 to +0.52 PR-AUC, robust across models, tasks,
and seeds); under honest temporal evaluation trees still beat GNNs and the post-shift
window is unsolved. **C2 (a working drift-robust method) remains elusive**: four
principled attempts (reconstruction, rolling-normal, unsupervised domain-adversarial,
supervised domain-adversarial) all fail. This is reported honestly - the value is a
rigorous reality check plus a mapped-out set of what does not work.

## Exp 13 - Generality round: MP-leak null everywhere, controls robust

- **Decomposition generalizes (5 seeds each).** GCN: base-rate +0.061, MP-leak
  +0.003 +/- 0.016, distribution access +0.489 (total +0.553). GAT: base-rate
  +0.073, MP-leak -0.010 +/- 0.033, distribution access +0.415 (total +0.478).
  With SAGE (0.000 +/- 0.007), the message-passing leakage channel is **null across
  all three architectures**; distribution access dominates everywhere.
- **Hetero inflation is seed-robust.** pre_split, seeds 42/43/44: inflation +0.497 /
  +0.597 / +0.551 -> **+0.548 +/- 0.041** (temporal 0.417 +/- 0.042, random
  0.966 +/- 0.002).
- **DGraph control holds with a wider model.** hidden 128: PR-AUC inflation +0.004
  (still ~null), ROC-AUC +0.055. The floor caveat stands; the control remains
  supporting evidence, not proof.
- **EvolveGCN sweep artifact** (`evolvegcn_sweep.json`): lr x clip grid, best test
  PR-AUC 0.200 (lr 0.005, no clip) vs 0.100 default - tuning helps somewhat but
  every configuration stays far below the honest baselines.

## Exp 14 - Published-recipe replication (5 seeds): the 0.98 reproduced exactly

**Setup.** `gnn-fraud recipe`: a generic XGBoost under the published recipe
(stratified random split + aggregate metrics) vs the honest protocol.

**Result (mean +/- sample std, n=5):**
| Metric | Recipe (random split) | Honest (temporal split) |
|---|---|---|
| Accuracy | **0.9917 +/- 0.0002** | 0.9692 +/- 0.0012 |
| Weighted F1 | **0.9916 +/- 0.0002** | 0.9688 +/- 0.0011 |
| Illicit F1 | 0.9564 +/- 0.0010 | 0.7559 +/- 0.0071 |
| PR-AUC | 0.9856 +/- 0.0012 | 0.7909 +/- 0.0015 |

**Conclusion.** The literature's ~0.98 headline (e.g. accuracy 0.9802 / weighted F1
0.9799) is reproduced and slightly exceeded by a generic tabular model under the
recipe - no novel architecture required. The table also shows the two tricks
separately: even under the honest temporal split, ACCURACY still reads 0.97 (the
majority class dominates), while the honest illicit-F1 is 0.76. Split choice and
metric choice each contribute to the illusion; together they turn 0.76 into 0.99.

**Post-decomposition update (Exp 10-12).** The central claim is now sharper and
partly self-corrected: the temporal-vs-random gap decomposes into a small base-rate
term, a **null** message-passing-leakage term (our own double-leak hypothesis,
refuted by the inductive ablation), and a dominant **distribution-access** term -
random splits let models train on the test period's distribution, and models that
generalize worst across the shift (GNNs) gain the most. All headline cells are now
multi-seed; the hetero conclusion survives the deployment-honest feature protocol.
This is the paper's core finding.
