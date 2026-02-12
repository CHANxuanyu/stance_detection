from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from .data import LABELS, build_dataset, load_rumoureval
from .metrics import compute_all_metrics, compute_subset_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RoBERTa stance classifier")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--no-target", action="store_true")
    p.add_argument("--no-replace-mentions-urls", action="store_true")
    p.add_argument("--subset-csv", default=None, help="CSV with columns id,target_dependent[,is_direct_reply]")
    p.add_argument("--save-preds", action="store_true")
    return p.parse_args()


def load_subset_map(path: str) -> Dict[str, Dict[str, bool]]:
    subset: Dict[str, Dict[str, bool]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "id" not in row:
                continue
            rid = str(row["id"])
            td_raw = row.get("target_dependent", row.get("target_dependency", row.get("targetDependent")))
            if td_raw is None:
                continue
            td = str(td_raw).strip().lower() in {"1", "true", "yes", "y"}
            direct_raw = row.get("is_direct_reply", row.get("direct_reply", row.get("is_direct")))
            is_direct = True if direct_raw is None else str(direct_raw).strip().lower() in {"1", "true", "yes", "y"}
            subset[rid] = {"target_dependent": td, "is_direct_reply": is_direct}
    return subset


def indices_for_subset(ids: List[str], subset_map: Dict[str, Dict[str, bool]], target_dependent: bool) -> List[int]:
    indices: List[int] = []
    for idx, rid in enumerate(ids):
        info = subset_map.get(str(rid))
        if info is None:
            continue
        if info.get("is_direct_reply") is False:
            continue
        if info.get("target_dependent") == target_dependent:
            indices.append(idx)
    return indices


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    raw_ds = load_rumoureval(cache_dir=args.cache_dir)
    tokenized = build_dataset(
        raw_ds,
        tokenizer=tokenizer,
        max_length=args.max_length,
        replace_urls_mentions=not args.no_replace_mentions_urls,
        use_target=not args.no_target,
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.model_dir, "_eval"),
        per_device_eval_batch_size=args.batch_size,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    preds_output = trainer.predict(tokenized["test"])
    logits = preds_output.predictions
    labels = preds_output.label_ids
    preds = np.argmax(logits, axis=-1)

    metrics = compute_all_metrics(labels.tolist(), preds.tolist())
    with open(os.path.join(args.model_dir, "metrics_full.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    subset_metrics = {}
    if args.subset_csv:
        subset_map = load_subset_map(args.subset_csv)
        ids = [str(x) for x in raw_ds["test"]["id"]]
        dep_idx = indices_for_subset(ids, subset_map, True)
        indep_idx = indices_for_subset(ids, subset_map, False)
        subset_metrics["target_dependent"] = compute_subset_metrics(labels.tolist(), preds.tolist(), dep_idx)
        subset_metrics["target_independent"] = compute_subset_metrics(labels.tolist(), preds.tolist(), indep_idx)
        with open(os.path.join(args.model_dir, "metrics_subsets.json"), "w", encoding="utf-8") as f:
            json.dump(subset_metrics, f, indent=2)

    if args.save_preds:
        out_path = os.path.join(args.model_dir, "predictions_test.csv")
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label", "pred"])
            for rid, y, p in zip(raw_ds["test"]["id"], labels, preds):
                writer.writerow([rid, LABELS[int(y)], LABELS[int(p)]])

    # print concise summary
    print("Full test set:")
    print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, ensure_ascii=False))
    if subset_metrics:
        print("Target-dependent subset:")
        print(json.dumps({k: round(v, 4) for k, v in subset_metrics["target_dependent"].items()}, ensure_ascii=False))
        print("Target-independent subset:")
        print(json.dumps({k: round(v, 4) for k, v in subset_metrics["target_independent"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
