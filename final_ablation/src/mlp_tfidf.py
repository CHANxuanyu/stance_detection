from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score


@dataclass
class MLPConfig:
    hidden_size: int = 128
    lr: float = 0.02
    epochs: int = 55
    batch_size: int = 64


class TfidfMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_labels: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, num_labels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.tanh(self.fc1(x))
        logits = self.fc2(hidden)
        return logits, hidden


class MLPTrainer:
    def __init__(self, config: MLPConfig, num_labels: int):
        self.config = config
        self.num_labels = num_labels

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> TfidfMLP:
        model = TfidfMLP(X_train.shape[1], self.config.hidden_size, self.num_labels)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.lr)
        loss_fct = nn.CrossEntropyLoss()

        best_f1 = -1.0
        best_state = None

        for epoch in range(self.config.epochs):
            model.train()
            perm = torch.randperm(X_train.size(0))
            for i in range(0, X_train.size(0), self.config.batch_size):
                idx = perm[i : i + self.config.batch_size]
                xb = X_train[idx]
                yb = y_train[idx]
                logits, _ = model(xb)
                loss = loss_fct(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits, _ = model(X_val)
                val_pred = torch.argmax(val_logits, dim=-1).cpu().numpy()
                f1 = f1_score(y_val.cpu().numpy(), val_pred, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        return model


def build_vectorizer() -> TfidfVectorizer:
    # Higher-capacity TF-IDF (bigrams + sublinear tf) often boosts deny/query
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        max_features=50000,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )


def save_mlp(model: TfidfMLP, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_mlp(path: str, input_dim: int, hidden_size: int, num_labels: int) -> TfidfMLP:
    model = TfidfMLP(input_dim, hidden_size, num_labels)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model


def save_config(config: MLPConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2)
