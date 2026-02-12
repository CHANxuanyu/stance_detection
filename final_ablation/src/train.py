from __future__ import annotations

import argparse
import json
import os
import random
from typing import Optional

import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from .data import LABELS, build_dataset, load_rumoureval
from .metrics import compute_all_metrics


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


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
    # normalize to keep scale stable
    weights = weights / weights.sum() * len(LABELS)
    return torch.tensor(weights, dtype=torch.float32)


def parse_args():
    p = argparse.ArgumentParser(description="Train RoBERTa-base stance classifier")
    p.add_argument("--model-name", default="roberta-base")
    p.add_argument("--output-dir", default="./outputs/roberta-base")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-power", type=float, default=0.0, help="class weight power; 0 disables weighting")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--no-target", action="store_true", help="use target-oblivious (reply only)")
    p.add_argument("--no-replace-mentions-urls", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    raw_ds = load_rumoureval(cache_dir=args.cache_dir)
    tokenized = build_dataset(
        raw_ds,
        tokenizer=tokenizer,
        max_length=args.max_length,
        replace_urls_mentions=not args.no_replace_mentions_urls,
        use_target=not args.no_target,
    )

    train_labels = raw_ds["train"]["label"]
    class_weights = None
    if args.weight_power and args.weight_power > 0:
        class_weights = compute_class_weights(train_labels, args.weight_power)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        save_total_limit=2,
        report_to=[],
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_all_metrics(labels.tolist(), preds.tolist())

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    train_result = trainer.train()
    trainer.save_model(args.output_dir)

    metrics = train_result.metrics
    with open(os.path.join(args.output_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # evaluate on validation and test
    eval_metrics = trainer.evaluate()
    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    test_metrics = trainer.evaluate(eval_dataset=tokenized["test"], metric_key_prefix="test")
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
