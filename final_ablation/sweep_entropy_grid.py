#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


def parse_csv_floats(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def run_one(cmd: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write("COMMAND:\n")
        logf.write(" ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=False)
    return proc.returncode


def read_metrics(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_float(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid search for residual entropy-gated fusion (train_proposed.py)"
    )
    parser.add_argument("--output-root", default="./outputs/sweeps/entropy_grid")
    parser.add_argument("--raw-dir", default="/home/chan/projects/stance_detection/RumourEval-2019-Stance-Detection/src")
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--fusion", default="entropy_gate", choices=["entropy_gate", "concat"])
    parser.add_argument("--gate-styles", default="residual,multiplicative")
    parser.add_argument("--weight-powers", default="0.5,0.55,0.6,0.65,0.7")
    parser.add_argument("--entropy-temperatures", default="0.6,0.7,0.8,0.9,1.0,1.1,1.4")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--mlp-epochs", type=int, default=55)
    parser.add_argument("--mlp-lr", type=float, default=0.02)
    parser.add_argument("--max-lengths", default="128,192,256")
    parser.add_argument("--tfidf-use-source", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip runs with existing metrics_full.json")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv = output_root / "summary.csv"

    gate_styles = parse_csv_strings(args.gate_styles)
    weight_powers = parse_csv_floats(args.weight_powers)
    entropy_temps = parse_csv_floats(args.entropy_temperatures)
    seeds = parse_csv_ints(args.seeds)
    max_lengths = parse_csv_ints(args.max_lengths)

    rows: List[Dict[str, str]] = []
    combos = list(itertools.product(gate_styles, weight_powers, entropy_temps, max_lengths, seeds))

    print(f"Total runs: {len(combos)}")
    for idx, (gate_style, weight_power, entropy_temp, max_length, seed) in enumerate(combos, start=1):
        run_name = (
            f"fusion-{args.fusion}_gate-{gate_style}_wp-{weight_power:g}"
            f"_tau-{entropy_temp:g}_len-{max_length}_seed-{seed}"
        )
        out_dir = output_root / run_name
        metrics_path = out_dir / "metrics_full.json"
        log_path = out_dir / "train.log"

        if args.resume and metrics_path.exists():
            metrics = read_metrics(metrics_path)
            status = "skipped_existing"
            code = 0
        else:
            cmd = [
                sys.executable,
                "-m",
                "src.train_proposed",
                "--model-name",
                args.model_name,
                "--output-dir",
                str(out_dir),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--mlp-hidden",
                str(args.mlp_hidden),
                "--mlp-epochs",
                str(args.mlp_epochs),
                "--mlp-lr",
                str(args.mlp_lr),
                "--max-length",
                str(max_length),
                "--use-raw",
                "--raw-dir",
                args.raw_dir,
                "--weight-power",
                str(weight_power),
                "--fusion",
                args.fusion,
                "--gate-style",
                gate_style,
                "--entropy-temperature",
                str(entropy_temp),
                "--seed",
                str(seed),
            ]
            if args.tfidf_use_source:
                cmd.append("--tfidf-use-source")

            print(f"[{idx}/{len(combos)}] {run_name}")
            code = run_one(cmd, log_path)
            metrics = read_metrics(metrics_path)
            status = "ok" if code == 0 and metrics else f"failed_{code}"

        row = {
            "run_name": run_name,
            "status": status,
            "seed": str(seed),
            "fusion": args.fusion,
            "gate_style": gate_style,
            "weight_power": str(weight_power),
            "entropy_temperature": str(entropy_temp),
            "max_length": str(max_length),
            "macro_f1": str(metrics.get("macro_f1", "")),
            "weighted_f2": str(metrics.get("weighted_f2", "")),
            "f2_support": str(metrics.get("f2_support", "")),
            "f2_deny": str(metrics.get("f2_deny", "")),
            "accuracy": str(metrics.get("accuracy", "")),
            "output_dir": str(out_dir),
        }
        rows.append(row)

    fieldnames = [
        "run_name",
        "status",
        "seed",
        "fusion",
        "gate_style",
        "weight_power",
        "entropy_temperature",
        "max_length",
        "macro_f1",
        "weighted_f2",
        "f2_support",
        "f2_deny",
        "accuracy",
        "output_dir",
    ]
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r["status"] in {"ok", "skipped_existing"} and r["macro_f1"]]
    ok_rows.sort(key=lambda r: float(r["macro_f1"]), reverse=True)

    print(f"\nSummary written: {summary_csv}")
    if not ok_rows:
        print("No successful runs with metrics.")
        return

    print("\nTop runs by macro_f1:")
    top = ok_rows[: args.top_k]
    for r in top:
        print(
            f"macro_f1={format_float(float(r['macro_f1']))} "
            f"wF2={format_float(float(r['weighted_f2']))} "
            f"F2(S)={format_float(float(r['f2_support']))} "
            f"F2(D)={format_float(float(r['f2_deny']))} "
            f"len={r['max_length']} "
            f"gate={r['gate_style']} wp={r['weight_power']} "
            f"tau={r['entropy_temperature']} seed={r['seed']}"
        )


if __name__ == "__main__":
    main()
