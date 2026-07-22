# Data Card

Provenance and license for every data source used in this project. Real, public
data only. Raw and processed files are **gitignored** - this card is how a reader
reproduces the data, not the data itself.

> Model results are never reported here. The statistics below the fold were
> **measured at ingestion (step 1)** with `gnn-fraud eda`, not copied from a
> paper.

## Primary: Elliptic Bitcoin dataset (via PyTorch Geometric)

- **What**: A graph of Bitcoin transactions labeled licit / illicit / unknown.
  Nodes = transactions, edges = payment flows, node features, 49 time steps.

### Verified statistics (measured with `gnn-fraud eda`, PyG build)

| Metric | Value |
|---|---|
| Nodes (transactions) | 203,769 |
| Edges | 234,355 |
| Node features (PyG build) | 165 |
| Directed | yes |
| Class: licit (`0`) | 42,019 |
| Class: illicit (`1`) | 4,545 |
| Class: unknown (`2`) | 157,205 |
| Labeled nodes (licit + illicit) | 46,564 |
| Illicit share of labeled | 9.76% |
| Illicit share of all nodes | 2.23% |
| Mean degree (in + out) | 2.30 |
| Max degree | 473 |
| Isolated nodes | 0 |
| Connected components | 49 |
| Largest component (fraction of nodes) | 3.87% |

Notes:
- PyG exposes **165** node features (the original release documents 166 = 94
  local + 72 aggregated + the time step; PyG's build yields 165 in `data.x`).
- **49 connected components for 49 time steps**: edges essentially do not cross
  time steps, so the graph is close to a disjoint union of per-time-step
  subgraphs. This matters for modeling (message passing stays within a time step).
- **Built-in temporal split** (PyG `train_mask` / `test_mask`): train has
  26,432 licit / 3,462 illicit; test has 15,587 licit / 1,083 illicit. We keep
  this split (early time steps -> train) and never randomize it.
- **Access**: `torch_geometric.datasets.EllipticBitcoinDataset` (and
  `EllipticBitcoinTemporalDataset`). Downloaded by PyG on first use. **No
  scraping.**
- **Original source**: Elliptic, released with the paper *"Anti-Money
  Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for
  Financial Forensics"* (Weber et al., 2019).
- **License / terms**: research use; the dataset was distributed publicly by
  Elliptic (originally via Kaggle). To confirm and pin the exact license text at
  ingestion. Cite the paper.
- **Why chosen**: packaged (reproducible in CI), temporal (enables a leakage-free
  split), a canonical benchmark (comparable to the literature), and clean enough
  to teach message passing on.

## Heterogeneous track: Elliptic++

- **What**: Extends Elliptic with **actors/addresses** in addition to
  transactions -> a genuinely heterogeneous graph (address and transaction node
  types). Fits the `HeteroData` / relational-schema framing (step 4).
- **Source**: GitHub repo [`git-disl/EllipticPlusPlus`](https://github.com/git-disl/EllipticPlusPlus);
  data hosted on the authors' Google Drive
  (folder id `1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l`).
- **Paper / citation (required)**: Elmougy & Liu, *"Demystifying Fraudulent
  Transactions and Illicit Nodes in the Bitcoin Network for Financial Forensics"*,
  **KDD '23**, DOI [`10.1145/3580305.3599803`](https://doi.org/10.1145/3580305.3599803).
- **Scale (from the paper/repo, to re-verify at ingestion)**:
  - Transactions: 203,769 nodes, 234,355 edges, 183 features.
  - Actors/addresses: 822,942 unique addresses (1,268,260 temporal interactions),
    56 features; Address-Address edges 2,868,964; Address-Transaction edges
    1,314,241.
- **License**: **No explicit license file is present in the repository and no
  formal terms of use are stated.** The data is shared publicly by the authors for
  research, with citation required. We use it for non-commercial research only,
  cite the KDD '23 paper, keep the raw data gitignored (never redistributed here),
  and will contact the authors (yelmougy3@gatech.edu) if formal terms are needed.
  This ambiguity is recorded deliberately rather than glossed over.
- **Status**: acquired for step 4 (user-approved); raw files in
  `data/raw/elliptic_pp/` (gitignored).

## Optional extension: real Ethereum on-chain data

Not required for the default pipeline; a track to demonstrate real ingestion.

- **Etherscan API**: free-tier key in `.env` (`ETHERSCAN_API_KEY`). Respect the
  documented rate limit (free tier is a few calls/sec). ToS-compliant, no scraping.
- **Google BigQuery public datasets** `crypto_ethereum` / `crypto_bitcoin`:
  query with a GCP project (`GCP_PROJECT_ID`). Public datasets, standard BigQuery
  terms; mind query cost.
- **Labels**: weak / to be sourced (e.g. published phishing address lists). Any
  labeling heuristic will be documented and never presented as ground truth.

## Comparison snapshot

| Dataset | Chain | Graph type | Labels | Temporal | Access | Effort |
|---|---|---|---|---|---|---|
| Elliptic (PyG) | Bitcoin | homogeneous (tx-tx) | licit/illicit/unknown | yes (49 steps) | PyG built-in | low |
| Elliptic++ | Bitcoin | heterogeneous (addr+tx) | on addresses & tx | yes | GitHub repo | medium |
| Ethereum phishing (XBlock) | Ethereum | directed multigraph | phishing/normal | partial | XBlock.pro | med-high |
| Etherscan / BigQuery | Ethereum | build-your-own | weak / derived | yes | official API | high |

## Integrity rules (recap)

- Provenance + license recorded here for anything we load.
- Raw/processed data never committed (see `.gitignore`).
- Secrets only in `.env`; `.env.example` documents the shape.
- Rate limits and ToS respected; no bypassing protections.
