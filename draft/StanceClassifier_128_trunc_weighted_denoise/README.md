# StanceClassifier
Stance Classifier for the WeVerify project, to determine the stance (support, deny, question, comment) of a "reply" tweet or other social media post towards the original "target" post to which it is replying.

> This is the latest version of the classifier, with target-aware and target-oblivious models based on BERTweet and XLM-RoBERTa.  For the older multilingual BERT-based models see the [bert-model branch](https://github.com/GateNLP/StanceClassifier/tree/bert-model)

## Available models

There are two versions of the stance classifier available:

- "target aware" - a mode that uses both the reply post _and_ the target post together to determine the stance.  This uses two models fine-tuned from [vinai/bertweet-base](https://huggingface.co/vinai/bertweet-base), one for just the reply and one for the target and reply together
- "target oblivious" - a mode that uses just the reply post, without reference to the target.  This model is fine tuned from [cardiffnlp/twitter-xlm-roberta-base](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base), which is a multilingual model, but the fine tuning data is still the RumourEval 2019 data set which is English only.

## Usage

### Basic usage
```
python -m StanceClassifier reply.json [original.json]
```

The reply and (if provided) original arguments should be JSON files containing tweets in JSON format with at least a `"text"` or `"full_text"` property containing the text.  If only a reply is provided then the target-oblivious model will be used, if an original tweet is provided as well then the ensemble model will be used that combines a target-oblivious and a target-aware model and picks the best classification.

The output is a class:
 - 0.0 = support
 - 1.0 = deny
 - 2.0 = query
 - 3.0 = comment

and a vector with the probabilities returned for each class.

The folder `examples` contains examples of original tweets and replies:
 - original_old and reply_old are examples of the old JSON files (140 characters)
 - original_new and reply_new are examples of the new JSON files (280 characters)

### Programmatic usage

The project provides three main classes, `StanceClassifer` for target-oblivious stance detection and `StanceClassifierEnsemble` for the target-aware ensemble model, plus `StanceClassifierWithTarget` that uses _only_ the target-aware model.  All these classes can be imported `from StanceClassifier.stance_classifier`, and will download their models from HuggingFace on first use.

### Server usage

The `docker` directory contains configuration to build a Docker image running a particular model of the classifier as an HTTP endpoint compliant with the [API specification](https://european-language-grid.readthedocs.io/en/release1.1.2/all/A2_API/LTInternalAPI.html) of the [European Language Grid](https://www.european-language-grid.eu).

### Training new models
To train new models, you can edit `train_model.py` (more support will be given in the future). To run:
```
python train_model.py
```

## Use local CSV dataset (train + predict)

For local datasets with columns `source_text`, `reply_text`, `label`, you can use:

### 1) Train on local CSV
```bash
python train_local_csv.py \
  --train-csv ./data/rumoureval_2019/rumoureval2019_train.csv \
  --val-csv ./data/rumoureval_2019/rumoureval2019_val.csv \
  --test-csv ./data/rumoureval_2019/rumoureval2019_test.csv \
  --base-model GateNLP/stance-bertweet-target-aware \
  --mode aware \
  --output-dir ./checkpoints/local-aware
```

If you want reply-only training, set `--mode oblivious`.

### 1b) Truncation optimization (fixed source/reply budgets, 128 tokens)
```bash
python train_local_csv.py \
  --train-csv ./data/rumoureval_2019/rumoureval2019_train.csv \
  --val-csv ./data/rumoureval_2019/rumoureval2019_val.csv \
  --test-csv ./data/rumoureval_2019/rumoureval2019_test.csv \
  --base-model GateNLP/stance-bertweet-target-aware \
  --mode aware \
  --max-length 128 \
  --use-pair-budget \
  --source-budget 32 \
  --reply-budget 92 \
  --reply-trunc-side head \
  --output-dir ./checkpoints/local-aware-budget32-92
```

### 1c) Macro-F1 optimization (class-weighted CE + truncation budgets)
```bash
python train_local_csv.py \
  --train-csv ./data/rumoureval_2019/rumoureval2019_train.csv \
  --val-csv ./data/rumoureval_2019/rumoureval2019_val.csv \
  --test-csv ./data/rumoureval_2019/rumoureval2019_test.csv \
  --base-model GateNLP/stance-bertweet-target-aware \
  --mode aware \
  --max-length 128 \
  --use-pair-budget \
  --source-budget 32 \
  --reply-budget 92 \
  --reply-trunc-side head \
  --loss-type weighted_ce \
  --class-weight-power 0.7 \
  --output-dir ./checkpoints/local-aware-budget32-92-wce-p07
```

### 1d) Sweep class-weight power automatically
```bash
python sweep_class_weight_power.py \
  --powers 0.5,0.7,1.0 \
  --summary-csv ./checkpoints/class_weight_power_sweep_summary.csv
```

It will launch multiple runs, save each full log as `*.log`, and write a summary CSV with
`val_macro_f1/test_macro_f1` for quick comparison.

### 1e) Input denoising on top of weighted baseline
```bash
python train_local_csv.py \
  --train-csv ./data/rumoureval_2019/rumoureval2019_train.csv \
  --val-csv ./data/rumoureval_2019/rumoureval2019_val.csv \
  --test-csv ./data/rumoureval_2019/rumoureval2019_test.csv \
  --base-model GateNLP/stance-bertweet-target-aware \
  --mode aware \
  --use-denoise \
  --max-length 128 \
  --use-pair-budget \
  --source-budget 32 \
  --reply-budget 92 \
  --reply-trunc-side head \
  --loss-type weighted_ce \
  --class-weight-power 0.7 \
  --output-dir ./checkpoints/local-aware-budget32-92-wce-p07-denoise
```

### 2) Predict on local CSV
```bash
python predict_local_csv.py \
  --model ./checkpoints/local-aware \
  --input-csv ./data/rumoureval_2019/rumoureval2019_test.csv \
  --output-csv ./predictions_test.csv \
  --mode aware
```

If `label` exists in the CSV, the script also prints metrics (Accuracy/Macro-F1/report).
