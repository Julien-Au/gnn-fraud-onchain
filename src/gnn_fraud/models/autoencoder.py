"""An unsupervised autoencoder baseline - a deliberate nod to USAD.

The USAD instinct: train a reconstruction model on *normal* data only, then score
anomalies by how badly they reconstruct. Here a plain symmetric MLP autoencoder
is trained on licit (normal) training nodes; the per-node reconstruction MSE is
the anomaly score. This is the honest, non-adversarial core of USAD; the
adversarial two-decoder refinement that made USAD strong on time series is noted
as a future variant, not claimed here.

No illicit labels are used to fit this model - it shows what is detectable
without knowing what fraud looks like.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn


class MLPAutoencoder(nn.Module):
    """Symmetric MLP autoencoder: in_dim -> 64 -> 32 -> 64 -> in_dim."""

    def __init__(self, in_dim: int, hidden1: int = 64, hidden2: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_autoencoder(
    x_normal: NDArray[Any],
    epochs: int = 60,
    lr: float = 1e-3,
    seed: int = 42,
) -> MLPAutoencoder:
    """Train the autoencoder on normal (licit) rows only. Fully seeded."""
    torch.manual_seed(seed)
    x = torch.from_numpy(np.asarray(x_normal, dtype=np.float32))
    model = MLPAutoencoder(in_dim=x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x), x)
        loss.backward()
        opt.step()
    return model


def reconstruction_error(model: MLPAutoencoder, x: NDArray[Any]) -> NDArray[Any]:
    """Per-row mean squared reconstruction error (the anomaly score)."""
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
        recon = model(xt)
        err = ((recon - xt) ** 2).mean(dim=1)
    return err.cpu().numpy()
