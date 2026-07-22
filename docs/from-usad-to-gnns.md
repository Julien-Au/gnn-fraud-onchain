# From temporal autoencoders (USAD) to GNNs on relational data

*A short narrative connecting my prior research to graph / relational learning.
Fleshed out as the project produces real results; this is the skeleton.*

## The through-line

USAD (KDD 2020) detects anomalies in **multivariate time series** without labels,
using two autoencoders trained adversarially so that one learns to amplify the
reconstruction error of inputs the other reconstructs too easily. The core ideas
carry directly into this project:

- **Anomaly = what the model cannot reconstruct / explain.** The unsupervised
  autoencoder baseline (step 2) is the same instinct applied to node features on a
  graph - a deliberate "a la USAD" reference point before any GNN.
- **Structure matters.** USAD models temporal dependence between sensors. Fraud on
  a blockchain lives in *relational* structure between accounts/transactions.
  GNNs generalize the "learn from neighbors" idea from a time axis to an arbitrary
  graph.

## What is genuinely new here (the gap I am closing)

- **Message passing** instead of a fixed temporal window: a node's representation
  aggregates its neighbors' (step 3 note in `LEARNING_NOTES.md`).
- **Heterogeneous / relational schemas**: multiple node and edge types
  (addresses, transactions) - the bridge to the "one model across many schemas"
  question a relational foundation model asks (step 4).

## The pitch (to be backed by results)

*How would this generalize toward a single model over many relational schemas?*
The ingestion layer is built schema-agnostic (PyG `HeteroData`), so a model is
parameterized by the schema rather than hard-coded to one dataset. This section
will summarize what the experiments actually show once steps 3-4 land - no claims
before real numbers.
