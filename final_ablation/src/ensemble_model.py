from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModel


class ProposedEnsemble(nn.Module):
    def __init__(
        self,
        roberta_name: str,
        mlp: nn.Module,
        hidden_size: int,
        num_labels: int,
        fusion: str = "entropy_gate",
        gate_style: str = "residual",
        entropy_temperature: float = 1.0,
    ):
        super().__init__()
        # Disable pooler to avoid random init warnings for RoBERTa
        self.roberta = AutoModel.from_pretrained(roberta_name, add_pooling_layer=False)
        self.mlp = mlp
        self.num_labels = num_labels
        self.fusion = fusion
        self.gate_style = gate_style
        self.entropy_temperature = entropy_temperature
        for p in self.mlp.parameters():
            p.requires_grad = False
        # Used only for uncertainty-aware gating from RoBERTa features.
        self.roberta_head = nn.Linear(self.roberta.config.hidden_size, num_labels)
        self.gate_scale = nn.Parameter(torch.tensor(4.0))
        self.gate_bias = nn.Parameter(torch.tensor(0.5))
        self.classifier = nn.Linear(self.roberta.config.hidden_size + hidden_size, num_labels)

    def _entropy_gate(self, pooled: torch.Tensor) -> torch.Tensor:
        roberta_logits = self.roberta_head(pooled)
        probs = torch.softmax(roberta_logits / self.entropy_temperature, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        norm_entropy = entropy / math.log(self.num_labels)
        alpha = torch.sigmoid(self.gate_scale * (norm_entropy - self.gate_bias))
        return alpha.unsqueeze(-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tfidf: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]

        with torch.no_grad():
            _, hidden = self.mlp(tfidf)

        if self.fusion == "entropy_gate":
            alpha = self._entropy_gate(pooled)
            if self.gate_style == "residual":
                hidden = (1.0 + alpha) * hidden
            else:
                hidden = alpha * hidden
        combined = torch.cat([pooled, hidden], dim=-1)
        logits = self.classifier(combined)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return logits, loss
