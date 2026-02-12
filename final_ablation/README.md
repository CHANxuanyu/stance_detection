# RumourEval-2019 RoBERTa Baseline + Proposed Ensemble (With Residual Entropy Gate)

This folder contains two pipelines:
1) **RoBERTa-base baseline** (standard fine-tuning)
2) **Paper proposed model**: RoBERTa pooled output + hidden layer of a **pretrained TF‑IDF MLP**, concatenated and classified
3) **TF‑IDF tuned variant**: (1,2)-grams + `sublinear_tf` + `max_features=50k` + `max_df=0.95`
4) **Original extension**: uncertainty-aware `entropy_gate` with residual scaling for RoBERTa/TF-IDF fusion

It computes the metrics you asked for, including `wF2` (weighted F2) and class‑specific `F2(S)` / `F2(D)`.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- The dataset is loaded from HuggingFace Hub, so network access is required the first time.
- This repo pins `datasets==2.19.1` because the dataset is script-based.
- If you want CPU-only PyTorch, install with:
  ```bash
  pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
  ```

## 1) Baseline (RoBERTa-base)

```bash
python -m src.train \
  --model-name roberta-base \
  --output-dir ./outputs/roberta-base \
  --epochs 3 \
  --batch-size 8 \
  --eval-batch-size 16
```

## 2) Proposed Ensemble + Entropy Gate

This follows the paper:
- TF‑IDF → MLP (hidden=128, tanh, LR=0.02, 55 epochs)
- RoBERTa pooled output + **MLP hidden layer** fusion
- Train ensemble for 6 epochs, LR=2e‑6, batch size 4

This folder defaults to `--fusion entropy_gate --gate-style residual`:
- Estimate RoBERTa uncertainty with normalized prediction entropy
- Learn a scalar gate `alpha` and scale TF-IDF hidden features as `(1 + alpha) * h_tfidf`
- Fused vector is `[h_roberta ; (1 + alpha) * h_tfidf]`

```bash
python -m src.train_proposed \
  --model-name roberta-base \
  --output-dir ./outputs/proposed_roberta_base \
  --epochs 6 \
  --batch-size 4 \
  --eval-batch-size 8 \
  --learning-rate 2e-6 \
  --mlp-hidden 128 \
  --mlp-epochs 55 \
  --mlp-lr 0.02
```

Optional flags:
- `--fusion entropy_gate|concat` to switch between your extension and paper-style concatenation
- `--gate-style residual|multiplicative` (recommended: `residual`)
- `--entropy-temperature 1.0` (tune to `0.8~1.5` for stability)
- `--no-target` for target‑oblivious (reply only)
- `--weight-power 0.7` to enable class weighting (cost‑weighting)
- `--tfidf-use-source` to use reply+source in TF‑IDF (default: reply only)
- `--use-raw --raw-dir /home/chan/projects/stance_detection/RumourEval-2019-Stance-Detection/src` to use raw RumourEval-2019 CSVs with **parent concatenation**

## 3) TF‑IDF tuned variant (this folder)

This folder uses a stronger TF‑IDF setup by default:
- `ngram_range=(1,2)`
- `sublinear_tf=True`
- `max_features=50000`
- `max_df=0.95`

Artifacts in `outputs/proposed_roberta_base`:
- `ensemble_model.pt`
- `tfidf_vectorizer.joblib`
- `mlp_tfidf.pt`
- `metrics_full.json`

## Evaluate proposed model

```bash
python -m src.eval_proposed \
  --model-dir ./outputs/proposed_roberta_base \
  --save-preds
```

This writes:
- `metrics_full.json`
- `predictions_test.csv`

### Target-dependent / independent subsets

The RumourEval‑2019 HF dataset does **not** include target‑dependence labels. If you have a CSV with those labels, you can pass it to compute the table in your screenshot.

Expected CSV format:

```csv
id,target_dependent,is_direct_reply
0,1,1
1,0,1
...
```

Then run:

```bash
python -m src.eval_proposed \
  --model-dir ./outputs/proposed_roberta_base \
  --subset-csv path/to/subset.csv
```

If you trained with `--use-raw`, evaluate with the same flags:

```bash
python -m src.eval_proposed \
  --model-dir ./outputs/proposed_roberta_base \
  --use-raw \
  --raw-dir /home/chan/projects/stance_detection/RumourEval-2019-Stance-Detection/src
```

It will produce:
- `metrics_subsets.json`

## Metrics reported

Full metrics include:
- `macro_f1`, `micro_f1`, `weighted_f1`
- `macro_f2`, `weighted_f2` (this is your `wF2`)
- Per‑class `f1_*` and `f2_*`, e.g. `f2_support` and `f2_deny`

## Grid Search (Recommended)

Run a grid search over residual entropy-gate parameters:

```bash
python sweep_entropy_grid.py \
  --output-root ./outputs/sweeps/entropy_grid_residual \
  --fusion entropy_gate \
  --gate-styles residual \
  --weight-powers 0.5,0.55,0.6,0.65,0.7 \
  --entropy-temperatures 0.6,0.7,0.8,0.9,1.0,1.1,1.4 \
  --max-lengths 128,192,256 \
  --seeds 42,43,44,45,46 \
  --resume
```

Outputs:
- `summary.csv` with all runs and metrics
- one subfolder per run containing `train.log` and `metrics_full.json`
