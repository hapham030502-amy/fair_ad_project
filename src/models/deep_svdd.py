from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class DeepSVDD(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_deep_svdd(X_train: np.ndarray, epochs: int = 20, lr: float = 1e-3, seed: int = 42):
    torch.manual_seed(seed)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    model = DeepSVDD(X_train_t.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    with torch.no_grad():
        center = model(X_train_t).mean(dim=0)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        z = model(X_train_t)
        loss = ((z - center) ** 2).mean()
        loss.backward()
        optimizer.step()

    return model, center


def score_deep_svdd(model, center, X: np.ndarray) -> np.ndarray:
    X_t = torch.tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        z = model(X_t)
        scores = ((z - center) ** 2).mean(dim=1).cpu().numpy()
    return scores
