#!/usr/bin/env python3
"""
Sweep class_weight_power for train_local_csv.py and summarize macro-F1.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run multiple trainings with different --class-weight-power values"
    )
    parser.add_argument("--workdir", default=str(here), help="Working directory to run training in")
    parser.add_argument("--train-script", default="train_local_csv.py", help="Training script path")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable")
    parser.add_argument("--powers", default="0.5,0.7,1.0", help="Comma-separated powers")
    parser.add_argument(
        "--output-prefix",
        default="./checkpoints/local-aware-budget32-92-wce",
        help="Per-run output dir prefix",
    )
    parser.add_argument(
        "--summary-csv",
        default="./checkpoints/class_weight_power_sweep_summary.csv",
        help="Path to save summary CSV",
    )
    parser.add_argument(
        "--train-csv",
        default="./data/rumoureval_2019/rumoureval2019_train.csv",
        help="Train CSV path",
    )
    parser.add_argument(
        "--val-csv",
        default="./data/rumoureval_2019/rumoureval2019_val.csv",
        help="Validation CSV path",
    )
    parser.add_argument(
        "--test-csv",
        default="./data/rumoureval_2019/rumoureval2019_test.csv",
        help="Test CSV path",
    )
    parser.add_argument("--base-model", default="GateNLP/stance-bertweet-target-aware")
    parser.add_argument("--mode", choices=["aware", "oblivious"], default="aware")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--source-budget", type=int, default=32)
    parser.add_argument("--reply-budget", type=int, default=92)
    parser.add_argument("--reply-trunc-side", choices=["head", "tail"], default="head")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_powers(raw: str) -> List[float]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("No valid powers parsed from --powers")
    return vals


def power_tag(value: float) -> str:
    txt = f"{value:.4f}".rstrip("0").rstrip(".")
    return txt.replace("-", "m").replace(".", "p")


def parse_metrics_from_log(text: str, prefix: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    pattern = re.compile(rf"^\s*{prefix}_(\w+):\s*([-+eE0-9\.]+)\s*$")
    for line in text.splitlines():
        m = pattern.match(line)
        if not m:
            continue
        key = f"{prefix}_{m.group(1)}"
        try:
            metrics[key] = float(m.group(2))
        except ValueError:
            continue
    return metrics


def run_one(args: argparse.Namespace, power: float) -> Dict[str, object]:
    tag = power_tag(power)
    out_dir = f"{args.output_prefix}-p{tag}"
    log_path = f"{out_dir}.log"
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    cmd = [
        args.python_bin,
        args.train_script,
        "--train-csv",
        args.train_csv,
        "--val-csv",
        args.val_csv,
        "--test-csv",
        args.test_csv,
        "--base-model",
        args.base_model,
        "--mode",
        args.mode,
        "--max-length",
        str(args.max_length),
        "--use-pair-budget",
        "--source-budget",
        str(args.source_budget),
        "--reply-budget",
        str(args.reply_budget),
        "--reply-trunc-side",
        args.reply_trunc_side,
        "--loss-type",
        "weighted_ce",
        "--class-weight-power",
        str(power),
        "--epochs",
        str(args.epochs),
        "--train-batch-size",
        str(args.train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--seed",
        str(args.seed),
        "--logging-steps",
        str(args.logging_steps),
        "--output-dir",
        out_dir,
    ]
    if args.no_fp16:
        cmd.append("--no-fp16")

    print(f"\n[RUN] class_weight_power={power}")
    print("[CMD] " + " ".join(cmd))
    print(f"[LOG] {log_path}")
    if args.dry_run:
        return {
            "power": power,
            "return_code": 0,
            "output_dir": out_dir,
            "log_file": log_path,
        }

    all_lines: List[str] = []
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=args.workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
            all_lines.append(line)
        return_code = proc.wait()

    log_text = "".join(all_lines)
    val_metrics = parse_metrics_from_log(log_text, "val")
    test_metrics = parse_metrics_from_log(log_text, "test")
    row: Dict[str, object] = {
        "power": power,
        "return_code": return_code,
        "output_dir": out_dir,
        "log_file": log_path,
    }
    row.update(val_metrics)
    row.update(test_metrics)
    return row


def write_summary(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = [
        "power",
        "return_code",
        "val_macro_f1",
        "test_macro_f1",
        "val_accuracy",
        "test_accuracy",
        "val_loss",
        "test_loss",
        "output_dir",
        "log_file",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def print_leaderboard(rows: List[Dict[str, object]]) -> None:
    ok_rows = [r for r in rows if r.get("return_code") == 0]
    ok_rows.sort(key=lambda r: float(r.get("val_macro_f1", -1.0)), reverse=True)
    print("\n=== Sweep Summary (sorted by val_macro_f1) ===")
    for row in ok_rows:
        p = row.get("power")
        vm = row.get("val_macro_f1", "")
        tm = row.get("test_macro_f1", "")
        va = row.get("val_accuracy", "")
        ta = row.get("test_accuracy", "")
        out = row.get("output_dir", "")
        print(
            f"power={p:<5} val_macro_f1={vm} test_macro_f1={tm} "
            f"val_acc={va} test_acc={ta} out={out}"
        )

    bad_rows = [r for r in rows if r.get("return_code") != 0]
    for row in bad_rows:
        print(
            f"power={row.get('power')} failed(return_code={row.get('return_code')}) "
            f"log={row.get('log_file')}"
        )


def main() -> None:
    args = parse_args()
    powers = parse_powers(args.powers)
    rows = []
    for power in powers:
        rows.append(run_one(args, power))

    write_summary(args.summary_csv, rows)
    print(f"\n[OK] summary saved to {args.summary_csv}")
    print_leaderboard(rows)


if __name__ == "__main__":
    main()

