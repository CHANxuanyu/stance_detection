from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

LABELS = ["support", "deny", "query", "comment"]
LABEL_TO_ID = {k: i for i, k in enumerate(LABELS)}


@dataclass
class SplitData:
    ids: List[str]
    reply_texts: List[str]
    source_texts: List[str]
    labels: List[int]


def _build_sequences(df: pd.DataFrame) -> SplitData:
    df = df.copy()
    df["text_x"] = df["text_x"].fillna("")
    df["inreText"] = df["inreText"].fillna("")
    df["sourceText"] = df["sourceText"].fillna("")

    ids = []
    reply_texts = []
    source_texts = []
    labels = []

    for _, row in df.iterrows():
        reply = str(row["text_x"])
        parent = str(row["inreText"])
        source = str(row["sourceText"])

        if source.strip() == "":
            # direct reply (parent is source) or source post
            seq1 = reply
            seq2 = parent.strip()
        else:
            # nested reply: concatenate parent to reply
            seq1 = (reply + " " + parent).strip() if parent.strip() else reply
            seq2 = source.strip()

        ids.append(str(row["id"]))
        reply_texts.append(seq1)
        source_texts.append(seq2)
        labels.append(LABEL_TO_ID[str(row["label_x"])])

    return SplitData(ids=ids, reply_texts=reply_texts, source_texts=source_texts, labels=labels)


def load_raw_rumoureval(raw_dir: str) -> Dict[str, SplitData]:
    raw = Path(raw_dir)
    train_files = [raw / "TwitterTrainDataSrc.csv", raw / "RedditTrainDataSrc.csv"]
    dev_files = [raw / "TwitterDevDataSrc.csv", raw / "RedditDevDataSrc.csv"]
    test_files = [raw / "TwitterTestDataSrc.csv", raw / "RedditTestDataSrc.csv"]

    def read_concat(files: List[Path]) -> pd.DataFrame:
        dfs = []
        for f in files:
            if f.exists():
                dfs.append(pd.read_csv(f))
        if not dfs:
            raise FileNotFoundError("No CSV files found in raw_dir: " + str(raw))
        return pd.concat(dfs, ignore_index=True)

    train_df = read_concat(train_files)
    dev_df = read_concat(dev_files)
    test_df = read_concat(test_files)

    return {
        "train": _build_sequences(train_df),
        "validation": _build_sequences(dev_df),
        "test": _build_sequences(test_df),
    }
