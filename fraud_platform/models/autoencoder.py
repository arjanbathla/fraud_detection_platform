"""Autoencoder (PyTorch) — reconstruction-error anomaly detector.

Trained on legitimate transactions only. It learns to reconstruct "normal" rows; fraud rows
reconstruct poorly, so per-row mean-squared reconstruction error is the anomaly score
(higher = more anomalous). Unsupervised in the sense that it never sees fraud during fit —
we only use the labels to select the normal rows to train on.

Small dense AE, trained on CPU. No GPU required for this data size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from fraud_platform.config import RANDOM_SEED
from fraud_platform.models.base import AnomalyModel


class _AENet(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderModel(AnomalyModel):
    name = "autoencoder"
    supervised = False

    def __init__(self, epochs: int = 30, batch_size: int = 256, lr: float = 1e-3,
                 latent_dim: int = 8):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.latent_dim = latent_dim
        self.net: _AENet | None = None
        self.input_dim: int | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "AutoencoderModel":
        torch.manual_seed(RANDOM_SEED)
        Xt = self.pipeline.fit_transform(X)
        # train on normal rows only if labels are available
        if y is not None:
            y = np.asarray(y)
            Xt_train = Xt[y == 0]
        else:
            Xt_train = Xt

        self.input_dim = Xt.shape[1]
        self.net = _AENet(self.input_dim, self.latent_dim)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        data = torch.tensor(Xt_train, dtype=torch.float32)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data),
            batch_size=self.batch_size, shuffle=True,
        )
        self.net.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                opt.zero_grad()
                out = self.net(batch)
                loss = loss_fn(out, batch)
                loss.backward()
                opt.step()

        # threshold: 99th percentile of reconstruction error on the training (normal) data
        train_err = self._recon_error(data)
        self.threshold = float(np.quantile(train_err, 0.99))
        return self

    def _recon_error(self, tensor: torch.Tensor) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            out = self.net(tensor)
            err = ((out - tensor) ** 2).mean(dim=1)
        return err.cpu().numpy()

    def score(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self._transform(X)
        tensor = torch.tensor(Xt, dtype=torch.float32)
        return self._recon_error(tensor)

    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.net.state_dict(),
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "pipeline": self.pipeline,
            "threshold": self.threshold,
        }, path)

    @classmethod
    def load(cls, path: str) -> "AutoencoderModel":
        blob = torch.load(path, weights_only=False)
        obj = cls(latent_dim=blob["latent_dim"])
        obj.input_dim = blob["input_dim"]
        obj.net = _AENet(obj.input_dim, obj.latent_dim)
        obj.net.load_state_dict(blob["state_dict"])
        obj.net.eval()
        obj.pipeline = blob["pipeline"]
        obj.threshold = blob["threshold"]
        return obj
