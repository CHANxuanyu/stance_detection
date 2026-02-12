#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
. .venv/bin/activate

python -m src.train_proposed \
  --model-name roberta-base \
  --output-dir ./outputs/final_best_entropy_wp055_tau12_seed44 \
  --epochs 6 \
  --batch-size 4 \
  --eval-batch-size 8 \
  --learning-rate 2e-6 \
  --mlp-hidden 128 \
  --mlp-epochs 55 \
  --mlp-lr 0.02 \
  --use-raw \
  --raw-dir /home/chan/projects/stance_detection/RumourEval-2019-Stance-Detection/src \
  --weight-power 0.55 \
  --fusion entropy_gate \
  --gate-style residual \
  --entropy-temperature 1.2 \
  --max-length 192 \
  --seed 44
