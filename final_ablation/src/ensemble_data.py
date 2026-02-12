from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .utils import normalize_text


@dataclass
class Example:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    tfidf: torch.Tensor
    label: int
    rid: str


class EnsembleDataset(Dataset):
    def __init__(
        self,
        ids: List[str],
        reply_texts: List[str],
        source_texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizerBase,
        tfidf_matrix,
        max_length: int = 256,
        use_target: bool = True,
        replace_urls_mentions: bool = True,
    ):
        self.ids = ids
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_target = use_target
        self.replace_urls_mentions = replace_urls_mentions
        self.reply_texts = reply_texts
        self.source_texts = source_texts
        self.tfidf_matrix = tfidf_matrix

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Example:
        reply = normalize_text(self.reply_texts[idx], self.replace_urls_mentions)
        source = normalize_text(self.source_texts[idx], self.replace_urls_mentions) if self.use_target else ""
        enc = self.tokenizer(
            reply,
            source,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        tfidf_row = self.tfidf_matrix[idx]
        if not isinstance(tfidf_row, np.ndarray):
            tfidf_row = tfidf_row.toarray()[0]
        tfidf_tensor = torch.tensor(tfidf_row, dtype=torch.float32)
        return Example(
            input_ids=enc["input_ids"].squeeze(0),
            attention_mask=enc["attention_mask"].squeeze(0),
            tfidf=tfidf_tensor,
            label=int(self.labels[idx]),
            rid=str(self.ids[idx]),
        )


def collate_fn(batch: List[Example], pad_token_id: int):
    input_ids = [b.input_ids for b in batch]
    attention_mask = [b.attention_mask for b in batch]
    tfidf = torch.stack([b.tfidf for b in batch])
    labels = torch.tensor([b.label for b in batch], dtype=torch.long)
    ids = [b.rid for b in batch]

    max_len = max(x.size(0) for x in input_ids)
    input_ids_padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attn_padded = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, (ids_i, attn_i) in enumerate(zip(input_ids, attention_mask)):
        input_ids_padded[i, : ids_i.size(0)] = ids_i
        attn_padded[i, : attn_i.size(0)] = attn_i

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attn_padded,
        "tfidf": tfidf,
        "labels": labels,
        "ids": ids,
    }
