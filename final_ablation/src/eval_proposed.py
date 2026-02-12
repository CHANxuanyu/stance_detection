from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List

import joblib
import torch
from transformers import AutoTokenizer

from .data import LABELS, load_rumoureval
from .ensemble_data import EnsembleDataset, collate_fn
from .ensemble_model import ProposedEnsemble
from .metrics import compute_all_metrics, compute_subset_metrics
from .mlp_tfidf import load_mlp
from .raw_rumoureval import load_raw_rumoureval
from .utils import normalize_text


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate proposed RoBERTa+TFIDF-MLP ensemble")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--subset-csv", default=None)
    p.add_argument("--save-preds", action="store_true")
    p.add_argument("--use-raw", action="store_true")
    p.add_argument("--raw-dir", default="/home/chan/projects/stance_detection/RumourEval-2019-Stance-Detection/src")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config
    with open(os.path.join(args.model_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    if args.use_raw:
        raw_splits = load_raw_rumoureval(args.raw_dir)
        test_replies = [normalize_text(t, config["replace_urls_mentions"]) for t in raw_splits["test"].reply_texts]
        test_sources = [normalize_text(t, config["replace_urls_mentions"]) for t in raw_splits["test"].source_texts]
        test_ids = raw_splits["test"].ids
        test_labels = raw_splits["test"].labels
    else:
        raw_ds = load_rumoureval(cache_dir=args.cache_dir)

        def prep_texts(split: str):
            replies = [normalize_text(t, config["replace_urls_mentions"]) for t in raw_ds[split]["reply_text"]]
            sources = [normalize_text(t, config["replace_urls_mentions"]) for t in raw_ds[split]["source_text"]]
            return replies, sources

        test_replies, test_sources = prep_texts("test")
        test_ids = raw_ds["test"]["id"]
        test_labels = raw_ds["test"]["label"]
    test_texts = [
        f"{r} {s}" if config["tfidf_use_source"] and s else r
        for r, s in zip(test_replies, test_sources)
    ]

    vectorizer = joblib.load(os.path.join(args.model_dir, "tfidf_vectorizer.joblib"))
    X_test = vectorizer.transform(test_texts)

    mlp_config_path = os.path.join(args.model_dir, "mlp_config.json")
    with open(mlp_config_path, "r", encoding="utf-8") as f:
        mlp_config = json.load(f)

    mlp = load_mlp(
        os.path.join(args.model_dir, "mlp_tfidf.pt"),
        input_dim=X_test.shape[1],
        hidden_size=mlp_config["hidden_size"],
        num_labels=len(LABELS),
    )

    model = ProposedEnsemble(
        config["model_name"],
        mlp,
        mlp_config["hidden_size"],
        len(LABELS),
        fusion=config.get("fusion", "concat"),
        gate_style=config.get("gate_style", "residual"),
        entropy_temperature=config.get("entropy_temperature", 1.0),
    )
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "ensemble_model.pt"), map_location="cpu"))
    model.to(device)
    model.eval()

    test_dataset = EnsembleDataset(
        ids=test_ids,
        reply_texts=test_replies,
        source_texts=test_sources,
        labels=test_labels,
        tokenizer=tokenizer,
        tfidf_matrix=X_test,
        max_length=config["max_length"],
        use_target=config["use_target"],
        replace_urls_mentions=config["replace_urls_mentions"],
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    all_preds = []
    all_labels = []
    all_ids = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tfidf = batch["tfidf"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(input_ids, attention_mask, tfidf, labels=None)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_ids.extend(batch["ids"])

    metrics = compute_all_metrics(all_labels, all_preds)
    with open(os.path.join(args.model_dir, "metrics_full.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    subset_metrics = {}
    if args.subset_csv:
        subset_map = load_subset_map(args.subset_csv)
        dep_idx = indices_for_subset(all_ids, subset_map, True)
        indep_idx = indices_for_subset(all_ids, subset_map, False)
        subset_metrics["target_dependent"] = compute_subset_metrics(all_labels, all_preds, dep_idx)
        subset_metrics["target_independent"] = compute_subset_metrics(all_labels, all_preds, indep_idx)
        with open(os.path.join(args.model_dir, "metrics_subsets.json"), "w", encoding="utf-8") as f:
            json.dump(subset_metrics, f, indent=2)

    if args.save_preds:
        out_path = os.path.join(args.model_dir, "predictions_test.csv")
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label", "pred"])
            for rid, y, p in zip(all_ids, all_labels, all_preds):
                writer.writerow([rid, LABELS[int(y)], LABELS[int(p)]])

    print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
