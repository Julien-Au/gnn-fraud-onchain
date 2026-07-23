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
