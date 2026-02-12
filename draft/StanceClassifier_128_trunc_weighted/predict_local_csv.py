#!/usr/bin/env python3
"""
Run stance prediction on a local CSV file.
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


FALLBACK_LABELS = ["support", "deny", "query", "comment"]


def parse_args():
    parser = argparse.ArgumentParser(description="Predict stance labels for local CSV")
    parser.add_argument("--model", required=True, help="HF model ID or local checkpoint path")
    parser.add_argument("--input-csv", required=True, help="Input CSV path")
    parser.add_argument("--output-csv", default="predictions.csv", help="Output CSV path")
    parser.add_argument(
        "--mode",
        choices=["aware", "oblivious"],
        default="aware",
        help="aware: source+reply input; oblivious: reply only",
    )
    parser.add_argument("--source-col", default="source_text", help="Source text column")
    parser.add_argument("--reply-col", default="reply_text", help="Reply text column")
    parser.add_argument("--label-col", default="label", help="Gold label column (optional)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max length")
    return parser.parse_args()


def load_label_names(model, num_labels):
    id2label = getattr(model.config, "id2label", None)
    if isinstance(id2label, dict) and len(id2label) == num_labels:
        result = []
        for i in range(num_labels):
            if i in id2label:
                result.append(str(id2label[i]))
            elif str(i) in id2label:
                result.append(str(id2label[str(i)]))
            else:
                result.append(f"LABEL_{i}")
        return result

    if num_labels == len(FALLBACK_LABELS):
        return FALLBACK_LABELS
    return [f"LABEL_{i}" for i in range(num_labels)]


def normalize_gold_labels(values, label_names):
    label2id = {name: i for i, name in enumerate(label_names)}
    lower_label2id = {name.lower(): i for name, i in label2id.items()}

    out = []
    for val in values:
        if isinstance(val, str):
            key = val.strip()
            if key in label2id:
                out.append(label2id[key])
                continue
            low = key.lower()
            if low in lower_label2id:
                out.append(lower_label2id[low])
                continue
            return None
        else:
            idx = int(val)
            if idx < 0 or idx >= len(label_names):
                return None
            out.append(idx)
    return np.array(out, dtype=np.int64)


@torch.inference_mode()
def predict(model, tokenizer, mode, sources, replies, batch_size, max_length, device):
    all_preds = []
    all_probs = []

    for i in range(0, len(replies), batch_size):
        batch_replies = replies[i : i + batch_size]
        if mode == "aware":
            batch_sources = sources[i : i + batch_size]
            enc = tokenizer(
                batch_sources,
                batch_replies,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
        else:
            enc = tokenizer(
                batch_replies,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )

        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        preds = probs.argmax(axis=-1)

        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    return np.array(all_preds, dtype=np.int64), np.array(all_probs, dtype=np.float32)


def main():
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    if args.reply_col not in df.columns:
        raise ValueError(f"Missing column '{args.reply_col}' in {args.input_csv}")
    if args.mode == "aware" and args.source_col not in df.columns:
        raise ValueError(f"Missing column '{args.source_col}' in {args.input_csv} for aware mode")

    replies = df[args.reply_col].fillna("").astype(str).tolist()
    sources = df[args.source_col].fillna("").astype(str).tolist() if args.mode == "aware" else [""] * len(replies)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    num_labels = model.config.num_labels
    label_names = load_label_names(model, num_labels)

    y_pred, y_prob = predict(
        model=model,
        tokenizer=tokenizer,
        mode=args.mode,
        sources=sources,
        replies=replies,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    df_out = df.copy()
    df_out["pred_id"] = y_pred
    df_out["pred_label"] = [label_names[i] for i in y_pred]
    for idx, name in enumerate(label_names):
        safe_name = str(name).replace(" ", "_")
        df_out[f"prob_{safe_name}"] = y_prob[:, idx]

    if args.label_col in df.columns:
        y_true = normalize_gold_labels(df[args.label_col].tolist(), label_names)
        if y_true is not None:
            macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            print(f"Accuracy : {acc:.4f}")
            print(f"Macro-F1 : {macro_f1:.4f}")
            print("\nClassification report:")
            print(
                classification_report(
                    y_true,
                    y_pred,
                    labels=list(range(len(label_names))),
                    target_names=label_names,
                    digits=4,
                    zero_division=0,
                )
            )
        else:
            print(
                f"Warning: column '{args.label_col}' exists but label values do not match model labels; "
                "skipping metrics."
            )

    df_out.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
