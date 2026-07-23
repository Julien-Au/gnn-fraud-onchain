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
