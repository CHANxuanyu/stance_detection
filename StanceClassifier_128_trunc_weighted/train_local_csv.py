#!/usr/bin/env python3
"""
Fine-tune a stance model on local CSV files.

Expected CSV columns by default:
  - source_text
  - reply_text
  - label   (string labels or numeric IDs)
"""

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


DEFAULT_LABELS = ["support", "deny", "query", "comment"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stance model from local CSV files")
    parser.add_argument("--train-csv", required=True, help="Path to train CSV")
    parser.add_argument("--val-csv", default=None, help="Path to validation CSV")
    parser.add_argument("--test-csv", default=None, help="Path to test CSV")
    parser.add_argument(
        "--base-model",
        default="GateNLP/stance-bertweet-target-aware",
        help="HF model ID or local checkpoint path",
    )
    parser.add_argument(
        "--mode",
        choices=["aware", "oblivious"],
        default="aware",
        help="aware: source+reply; oblivious: reply only",
    )
    parser.add_argument("--output-dir", default="./local_stance_model", help="Output model directory")
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated label order (defines label IDs)",
    )
    parser.add_argument("--source-col", default="source_text", help="Source text column name")
    parser.add_argument("--reply-col", default="reply_text", help="Reply text column name")
    parser.add_argument("--label-col", default="label", help="Label column name")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max length")
    parser.add_argument(
        "--use-pair-budget",
        action="store_true",
        help="Use fixed source/reply budgets in aware mode instead of generic pair truncation",
    )
    parser.add_argument(
        "--source-budget",
        type=int,
        default=32,
        help="Source token budget when --use-pair-budget is enabled",
    )
    parser.add_argument(
        "--reply-budget",
        type=int,
        default=None,
        help="Reply token budget when --use-pair-budget is enabled; defaults to remaining budget",
    )
    parser.add_argument(
        "--reply-trunc-side",
        choices=["head", "tail"],
        default="tail",
        help="When reply exceeds budget, keep head or tail",
    )
    parser.add_argument(
        "--truncation-strategy",
        choices=["longest_first", "only_first"],
        default="longest_first",
        help="Tokenizer truncation strategy when not using pair budget",
    )
    parser.add_argument("--epochs", type=float, default=3.0, help="Num epochs")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Train batch size per device")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Eval batch size per device")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--loss-type",
        choices=["ce", "weighted_ce"],
        default="weighted_ce",
        help="Loss type: standard CE or class-weighted CE",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=0.7,
        help="Weight strength for weighted CE. 1.0=inverse frequency, 0.5=sqrt inverse",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--logging-steps", type=int, default=50, help="Logging interval")
    parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 even on CUDA")
    return parser.parse_args()


class EncodedDataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Compute loss ourselves so we can inject class weights.
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is None:
            loss_fct = nn.CrossEntropyLoss()
        else:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))

        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def _normalize_label(value, label2id, lower_label2id) -> int:
    if pd.isna(value):
        raise ValueError("Missing label (NaN)")

    if isinstance(value, str):
        key = value.strip()
        if not key:
            raise ValueError("Missing label (empty string)")
        if key in label2id:
            return label2id[key]
        low = key.lower()
        if low in lower_label2id:
            return lower_label2id[low]

        # Also support stringified numeric labels, e.g. "0" / "0.0".
        try:
            num = float(key)
            if num.is_integer():
                idx = int(num)
                if 0 <= idx < len(label2id):
                    return idx
        except ValueError:
            pass

        raise ValueError(
            f"Unknown label '{value}'. Expected one of {list(label2id.keys())} or numeric IDs."
        )

    idx = int(value)
    if idx < 0 or idx >= len(label2id):
        raise ValueError(f"Label ID {idx} out of range [0, {len(label2id)-1}]")
    return idx


def load_split(
    csv_path: str,
    source_col: str,
    reply_col: str,
    label_col: str,
    label2id: dict,
) -> Tuple[List[str], List[str], List[int]]:
    resolved_csv_path = csv_path
    if not os.path.isfile(resolved_csv_path) and resolved_csv_path.startswith("../data/"):
        fallback = "./data/" + resolved_csv_path[len("../data/") :]
        if os.path.isfile(fallback):
            print(f"[INFO] input path fallback: {resolved_csv_path} -> {fallback}")
            resolved_csv_path = fallback
    if not os.path.isfile(resolved_csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path} (resolved: {resolved_csv_path}, cwd: {os.getcwd()})"
        )

    df = pd.read_csv(resolved_csv_path)
    for col in [source_col, reply_col, label_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {resolved_csv_path}. Found: {list(df.columns)}")

    missing_mask = df[label_col].isna() | (df[label_col].astype(str).str.strip() == "")
    missing_count = int(missing_mask.sum())
    if missing_count > 0:
        missing_rows = (df.index[missing_mask] + 2).tolist()[:10]  # +2 accounts for header + 0-index
        print(
            f"[WARN] {resolved_csv_path}: dropping {missing_count} rows with missing '{label_col}'. "
            f"Example CSV line numbers: {missing_rows}"
        )
        df = df.loc[~missing_mask].copy()
        if df.empty:
            raise ValueError(f"All rows in {resolved_csv_path} have missing '{label_col}'")

    sources = df[source_col].fillna("").astype(str).tolist()
    replies = df[reply_col].fillna("").astype(str).tolist()
    raw_labels = df[label_col].tolist()
    row_ids = df.index.tolist()

    lower_label2id = {k.lower(): v for k, v in label2id.items()}
    labels = []
    for row_idx, val in zip(row_ids, raw_labels):
        try:
            labels.append(_normalize_label(val, label2id, lower_label2id))
        except Exception as ex:
            raise ValueError(
                f"Invalid label at CSV line {row_idx + 2} in {resolved_csv_path}: {val!r}. {ex}"
            ) from ex
    return sources, replies, labels


def tokenize_split(
    tokenizer,
    mode: str,
    sources: List[str],
    replies: List[str],
    max_length: int,
    use_pair_budget: bool,
    source_budget: int,
    reply_budget: int,
    reply_trunc_side: str,
    truncation_strategy: str,
):
    if mode == "aware" and use_pair_budget:
        pair_special = tokenizer.num_special_tokens_to_add(pair=True)
        available = max_length - pair_special
        if available <= 1:
            raise ValueError(
                f"max_length={max_length} too small for pair input (special_tokens={pair_special})"
            )
        if source_budget < 0:
            raise ValueError(f"source_budget must be >= 0, got {source_budget}")

        if reply_budget is None:
            reply_budget = max(1, available - source_budget)
        if reply_budget < 1:
            raise ValueError(f"reply_budget must be >= 1, got {reply_budget}")

        # Fit budgets into available slots.
        if source_budget + reply_budget > available:
            overflow = source_budget + reply_budget - available
            reply_budget = max(1, reply_budget - overflow)
            if source_budget + reply_budget > available:
                source_budget = max(0, available - reply_budget)

        enc = {"input_ids": [], "attention_mask": []}
        include_token_type = "token_type_ids" in tokenizer.model_input_names
        if include_token_type:
            enc["token_type_ids"] = []

        src_clipped = 0
        rep_clipped = 0
        for src, rep in zip(sources, replies):
            src_ids = tokenizer.encode(str(src), add_special_tokens=False)
            rep_ids = tokenizer.encode(str(rep), add_special_tokens=False)

            if len(src_ids) > source_budget:
                src_clipped += 1
                src_ids = src_ids[:source_budget]

            if len(rep_ids) > reply_budget:
                rep_clipped += 1
                if reply_trunc_side == "tail":
                    rep_ids = rep_ids[-reply_budget:]
                else:
                    rep_ids = rep_ids[:reply_budget]

            input_ids = tokenizer.build_inputs_with_special_tokens(src_ids, rep_ids)
            enc["input_ids"].append(input_ids[:max_length])
            enc["attention_mask"].append([1] * min(len(input_ids), max_length))
            if include_token_type:
                tok_type = tokenizer.create_token_type_ids_from_sequences(src_ids, rep_ids)
                enc["token_type_ids"].append(tok_type[:max_length])

        print(
            f"[INFO] pair-budget enabled: source_budget={source_budget}, reply_budget={reply_budget}, "
            f"reply_trunc_side={reply_trunc_side}, source_clipped={src_clipped}, reply_clipped={rep_clipped}"
        )
        return enc

    if mode == "aware":
        return tokenizer(
            sources,
            replies,
            truncation=truncation_strategy,
            max_length=max_length,
            return_overflowing_tokens=False,
            verbose=False,
        )
    return tokenizer(
        replies,
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=False,
        verbose=False,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": precision,
        "macro_recall": recall,
    }


def compute_class_weights(train_labels: List[int], num_classes: int, power: float) -> torch.Tensor:
    counts = np.bincount(np.array(train_labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = (counts.sum() / counts) ** power
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def main():
    args = parse_args()

    labels = [lab.strip() for lab in args.labels.split(",") if lab.strip()]
    if not labels:
        raise ValueError("--labels is empty after parsing")

    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        ignore_mismatched_sizes=True,
    )
    model.config.label2id = label2id
    model.config.id2label = {str(k): v for k, v in id2label.items()}

    # Keep sequence length within model/tokenizer limits to avoid position embedding overflow.
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if tokenizer_limit is None or tokenizer_limit > 100000:
        tokenizer_limit = args.max_length
    model_limit = getattr(model.config, "max_position_embeddings", None)
    if model_limit is None:
        model_limit = args.max_length
    effective_max_length = int(min(args.max_length, tokenizer_limit, model_limit))
    if effective_max_length < 8:
        raise ValueError(
            f"Resolved max length too small ({effective_max_length}). "
            f"tokenizer_limit={tokenizer_limit}, model_limit={model_limit}"
        )
    if effective_max_length != args.max_length:
        print(
            f"[INFO] max_length adjusted: requested={args.max_length}, "
            f"tokenizer_limit={tokenizer_limit}, model_limit={model_limit}, "
            f"using={effective_max_length}"
        )

    tr_sources, tr_replies, tr_labels = load_split(
        args.train_csv, args.source_col, args.reply_col, args.label_col, label2id
    )
    tr_enc = tokenize_split(
        tokenizer,
        args.mode,
        tr_sources,
        tr_replies,
        effective_max_length,
        use_pair_budget=args.use_pair_budget,
        source_budget=args.source_budget,
        reply_budget=args.reply_budget,
        reply_trunc_side=args.reply_trunc_side,
        truncation_strategy=args.truncation_strategy,
    )
    train_ds = EncodedDataset(tr_enc, tr_labels)

    val_ds = None
    if args.val_csv:
        va_sources, va_replies, va_labels = load_split(
            args.val_csv, args.source_col, args.reply_col, args.label_col, label2id
        )
        va_enc = tokenize_split(
            tokenizer,
            args.mode,
            va_sources,
            va_replies,
            effective_max_length,
            use_pair_budget=args.use_pair_budget,
            source_budget=args.source_budget,
            reply_budget=args.reply_budget,
            reply_trunc_side=args.reply_trunc_side,
            truncation_strategy=args.truncation_strategy,
        )
        val_ds = EncodedDataset(va_enc, va_labels)

    test_ds = None
    if args.test_csv:
        te_sources, te_replies, te_labels = load_split(
            args.test_csv, args.source_col, args.reply_col, args.label_col, label2id
        )
        te_enc = tokenize_split(
            tokenizer,
            args.mode,
            te_sources,
            te_replies,
            effective_max_length,
            use_pair_budget=args.use_pair_budget,
            source_budget=args.source_budget,
            reply_budget=args.reply_budget,
            reply_trunc_side=args.reply_trunc_side,
            truncation_strategy=args.truncation_strategy,
        )
        test_ds = EncodedDataset(te_enc, te_labels)

    class_weights = None
    if args.loss_type == "weighted_ce":
        class_weights = compute_class_weights(
            train_labels=tr_labels,
            num_classes=len(labels),
            power=args.class_weight_power,
        )
        print(f"[INFO] loss_type=weighted_ce, class_weight_power={args.class_weight_power}")
        for idx, lab in enumerate(labels):
            print(f"  class_weight[{idx}] {lab}: {class_weights[idx].item():.4f}")
    else:
        print("[INFO] loss_type=ce")

    use_fp16 = (not args.no_fp16) and torch.cuda.is_available()

    if val_ds is not None:
        evaluation_strategy = "epoch"
        save_strategy = "epoch"
        load_best_model_at_end = True
    else:
        evaluation_strategy = "no"
        save_strategy = "epoch"
        load_best_model_at_end = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=args.logging_steps,
        report_to="none",
        seed=args.seed,
        fp16=use_fp16,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print(f"Training mode: {args.mode}")
    print(f"Train samples: {len(train_ds)}")
    if val_ds is not None:
        print(f"Val samples  : {len(val_ds)}")
    if test_ds is not None:
        print(f"Test samples : {len(test_ds)}")

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "label_mapping.json"), "w", encoding="utf-8") as out:
        json.dump(
            {
                "labels": labels,
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
                "mode": args.mode,
                "source_col": args.source_col,
                "reply_col": args.reply_col,
                "label_col": args.label_col,
                "use_pair_budget": args.use_pair_budget,
                "source_budget": args.source_budget,
                "reply_budget": args.reply_budget,
                "reply_trunc_side": args.reply_trunc_side,
                "truncation_strategy": args.truncation_strategy,
                "loss_type": args.loss_type,
                "class_weight_power": args.class_weight_power,
                "class_weights": class_weights.tolist() if class_weights is not None else None,
            },
            out,
            ensure_ascii=False,
            indent=2,
        )

    if val_ds is not None:
        metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="val")
        print("\nValidation metrics:")
        for key, value in sorted(metrics.items()):
            if key.startswith("val_"):
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    if test_ds is not None:
        metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
        print("\nTest metrics:")
        for key, value in sorted(metrics.items()):
            if key.startswith("test_"):
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print(f"\nSaved model to: {args.output_dir}")


if __name__ == "__main__":
    main()
