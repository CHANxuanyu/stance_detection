from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from .data import LABELS, load_rumoureval
from .ensemble_data import EnsembleDataset, collate_fn
from .ensemble_model import ProposedEnsemble
from .metrics import compute_all_metrics
from .mlp_tfidf import MLPConfig, MLPTrainer, build_vectorizer, save_config, save_mlp
from .raw_rumoureval import load_raw_rumoureval
from .utils import normalize_text


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels, power: float) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(LABELS)).astype(np.float32)
    inv = 1.0 / np.maximum(counts, 1.0)
    weights = np.power(inv, power)
    weights = weights / weights.sum() * len(LABELS)
    return torch.tensor(weights, dtype=torch.float32)


def parse_args():
    p = argparse.ArgumentParser(description="Train proposed RoBERTa+TFIDF-MLP ensemble")
    p.add_argument("--model-name", default="roberta-base")
    p.add_argument("--output-dir", default="./outputs/proposed_roberta_base")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=2e-6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--no-target", action="store_true")
    p.add_argument("--no-replace-mentions-urls", action="store_true")
    p.add_argument("--tfidf-use-source", action="store_true", help="use reply+source text for TF-IDF")
    p.add_argument("--weight-power", type=float, default=0.0, help="class weight power; 0 disables weighting")
    p.add_argument("--mlp-hidden", type=int, default=128)
    p.add_argument("--mlp-epochs", type=int, default=55)
    p.add_argument("--mlp-lr", type=float, default=0.02)
    p.add_argument("--fusion", choices=["concat", "entropy_gate"], default="entropy_gate")
    p.add_argument("--gate-style", choices=["residual", "multiplicative"], default="residual")
    p.add_argument("--entropy-temperature", type=float, default=1.0)
    p.add_argument("--use-raw", action="store_true", help="use raw RumourEval-2019 CSVs with parent concatenation")
    p.add_argument("--raw-dir", default="/home/chan/projects/stance_detection/RumourEval-2019-Stance-Detection/src")
    return p.parse_args()


def build_tfidf_text(reply: str, source: str, use_source: bool) -> str:
    if use_source and source:
        return f"{reply} {source}"
    return reply


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.use_raw:
        raw_splits = load_raw_rumoureval(args.raw_dir)
        train_replies = [normalize_text(t, not args.no_replace_mentions_urls) for t in raw_splits["train"].reply_texts]
        train_sources = [normalize_text(t, not args.no_replace_mentions_urls) for t in raw_splits["train"].source_texts]
        val_replies = [normalize_text(t, not args.no_replace_mentions_urls) for t in raw_splits["validation"].reply_texts]
        val_sources = [normalize_text(t, not args.no_replace_mentions_urls) for t in raw_splits["validation"].source_texts]
        test_replies = [normalize_text(t, not args.no_replace_mentions_urls) for t in raw_splits["test"].reply_texts]
        test_sources = [normalize_text(t, not args.no_replace_mentions_urls) for t in raw_splits["test"].source_texts]
    else:
        raw_ds = load_rumoureval(cache_dir=args.cache_dir)

        # Prepare texts
        def prep_texts(split: str):
            replies = [normalize_text(t, not args.no_replace_mentions_urls) for t in raw_ds[split]["reply_text"]]
            sources = [normalize_text(t, not args.no_replace_mentions_urls) for t in raw_ds[split]["source_text"]]
            return replies, sources

        train_replies, train_sources = prep_texts("train")
        val_replies, val_sources = prep_texts("validation")
        test_replies, test_sources = prep_texts("test")

    train_texts = [build_tfidf_text(r, s, args.tfidf_use_source) for r, s in zip(train_replies, train_sources)]
    val_texts = [build_tfidf_text(r, s, args.tfidf_use_source) for r, s in zip(val_replies, val_sources)]
    test_texts = [build_tfidf_text(r, s, args.tfidf_use_source) for r, s in zip(test_replies, test_sources)]

    # TF-IDF vectorizer
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    # MLP pretrain
    if args.use_raw:
        y_train = torch.tensor(raw_splits["train"].labels, dtype=torch.long)
        y_val = torch.tensor(raw_splits["validation"].labels, dtype=torch.long)
    else:
        y_train = torch.tensor(raw_ds["train"]["label"], dtype=torch.long)
        y_val = torch.tensor(raw_ds["validation"]["label"], dtype=torch.long)

    mlp_config = MLPConfig(hidden_size=args.mlp_hidden, lr=args.mlp_lr, epochs=args.mlp_epochs)
    mlp_trainer = MLPTrainer(mlp_config, num_labels=len(LABELS))
    mlp_model = mlp_trainer.fit(
        torch.tensor(X_train.toarray(), dtype=torch.float32),
        y_train,
        torch.tensor(X_val.toarray(), dtype=torch.float32),
        y_val,
    )

    # Save vectorizer + MLP
    import joblib

    joblib.dump(vectorizer, os.path.join(args.output_dir, "tfidf_vectorizer.joblib"))
    save_mlp(mlp_model, os.path.join(args.output_dir, "mlp_tfidf.pt"))
    save_config(mlp_config, os.path.join(args.output_dir, "mlp_config.json"))

    # Ensemble training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = EnsembleDataset(
        ids=(raw_splits["train"].ids if args.use_raw else raw_ds["train"]["id"]),
        reply_texts=train_replies,
        source_texts=train_sources,
        labels=(raw_splits["train"].labels if args.use_raw else raw_ds["train"]["label"]),
        tokenizer=tokenizer,
        tfidf_matrix=X_train,
        max_length=args.max_length,
        use_target=not args.no_target,
        replace_urls_mentions=not args.no_replace_mentions_urls,
    )
    val_dataset = EnsembleDataset(
        ids=(raw_splits["validation"].ids if args.use_raw else raw_ds["validation"]["id"]),
        reply_texts=val_replies,
        source_texts=val_sources,
        labels=(raw_splits["validation"].labels if args.use_raw else raw_ds["validation"]["label"]),
        tokenizer=tokenizer,
        tfidf_matrix=X_val,
        max_length=args.max_length,
        use_target=not args.no_target,
        replace_urls_mentions=not args.no_replace_mentions_urls,
    )
    test_dataset = EnsembleDataset(
        ids=(raw_splits["test"].ids if args.use_raw else raw_ds["test"]["id"]),
        reply_texts=test_replies,
        source_texts=test_sources,
        labels=(raw_splits["test"].labels if args.use_raw else raw_ds["test"]["label"]),
        tokenizer=tokenizer,
        tfidf_matrix=X_test,
        max_length=args.max_length,
        use_target=not args.no_target,
        replace_urls_mentions=not args.no_replace_mentions_urls,
    )

    mlp_model.eval()
    model = ProposedEnsemble(
        args.model_name,
        mlp_model,
        args.mlp_hidden,
        len(LABELS),
        fusion=args.fusion,
        gate_style=args.gate_style,
        entropy_temperature=args.entropy_temperature,
    ).to(device)

    if args.weight_power and args.weight_power > 0:
        base_labels = raw_splits["train"].labels if args.use_raw else raw_ds["train"]["label"]
        class_weights = compute_class_weights(base_labels, args.weight_power).to(device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fct = torch.nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_f1 = -1.0
    best_state = None

    def eval_loader(loader) -> Dict[str, float]:
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                tfidf = batch["tfidf"].to(device)
                labels = batch["labels"].to(device)
                logits, _ = model(input_ids, attention_mask, tfidf, labels=None)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        return compute_all_metrics(all_labels, all_preds)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tfidf = batch["tfidf"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(input_ids, attention_mask, tfidf, labels=None)
            loss = loss_fct(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step_count += 1
            progress.set_postfix({"loss": f"{epoch_loss / max(step_count,1):.4f}"})

        val_metrics = eval_loader(val_loader)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        with open(os.path.join(args.output_dir, "val_metrics_epoch.json"), "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, indent=2)

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "ensemble_model.pt"))
    tokenizer.save_pretrained(args.output_dir)

    # evaluate on test
    test_metrics = eval_loader(test_loader)
    with open(os.path.join(args.output_dir, "metrics_full.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # Save config
    config = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "use_target": not args.no_target,
        "replace_urls_mentions": not args.no_replace_mentions_urls,
        "tfidf_use_source": args.tfidf_use_source,
        "fusion": args.fusion,
        "gate_style": args.gate_style,
        "entropy_temperature": args.entropy_temperature,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(json.dumps({k: round(v, 4) for k, v in test_metrics.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
