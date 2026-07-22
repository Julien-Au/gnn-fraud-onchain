# Data Card

Provenance and license for every data source used in this project. Real, public
data only. Raw and processed files are **gitignored** - this card is how a reader
reproduces the data, not the data itself.

> Figures below are from the datasets' papers / repositories and will be
> **re-verified at ingestion** (step 1) and updated with the exact counts we
> observe. Nothing here is a reported model result.

## Primary: Elliptic Bitcoin dataset (via PyTorch Geometric)

- **What**: A graph of Bitcoin transactions labeled licit / illicit / unknown.
  Nodes = transactions, edges = payment flows, 166 node features (94 local +
  72 aggregated), 49 time steps.
- **Scale (to verify)**: ~203k transaction nodes, ~234k edges; ~23% of nodes
  labeled, of which a small minority (~2% of all nodes) are illicit -> strong
  class imbalance.
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
- **Access**: public research repository (GitHub). To pin the exact URL, commit
  hash, and license at ingestion.
- **Citation**: Elliptic++ dataset (Elmougy & Liu). Cite the paper/repo.
- **Status**: used from step 4.

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
