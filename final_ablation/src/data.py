from __future__ import annotations

from typing import Dict, Optional

from datasets import DatasetDict, load_dataset

from .utils import normalize_text

LABELS = ["support", "deny", "query", "comment"]


def load_rumoureval(cache_dir: Optional[str] = None) -> DatasetDict:
    ds = load_dataset("strombergnlp/rumoureval_2019", cache_dir=cache_dir)
    # Some splits may include missing labels; filter them out for training/eval consistency.
    def _has_label(example):
        return example.get("label") is not None

    for split in list(ds.keys()):
        ds[split] = ds[split].filter(_has_label)
    return ds


def build_dataset(
    datasets: DatasetDict,
    tokenizer,
    max_length: int = 256,
    replace_urls_mentions: bool = True,
    use_target: bool = True,
) -> DatasetDict:
    def _prepare(example: Dict):
        reply = normalize_text(example.get("reply_text"), replace_urls_mentions)
        source = normalize_text(example.get("source_text"), replace_urls_mentions) if use_target else ""
        encoded = tokenizer(
            reply,
            source,
            max_length=max_length,
            padding=False,
            truncation=True,
        )
        encoded["labels"] = example["label"]
        return encoded

    tokenized = datasets.map(_prepare, remove_columns=datasets["train"].column_names)
    tokenized.set_format(type="torch")
    return tokenized
