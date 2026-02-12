from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, fbeta_score


LABELS = ["support", "deny", "query", "comment"]


def compute_all_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    metrics["macro_f2"] = float(fbeta_score(y_true, y_pred, average="macro", beta=2, zero_division=0))
    metrics["weighted_f2"] = float(fbeta_score(y_true, y_pred, average="weighted", beta=2, zero_division=0))

    per_class_f1 = f1_score(y_true, y_pred, labels=list(range(len(LABELS))), average=None, zero_division=0)
    per_class_f2 = fbeta_score(y_true, y_pred, labels=list(range(len(LABELS))), average=None, beta=2, zero_division=0)

    for idx, label in enumerate(LABELS):
        metrics[f"f1_{label}"] = float(per_class_f1[idx])
        metrics[f"f2_{label}"] = float(per_class_f2[idx])

    return metrics


def compute_subset_metrics(y_true: List[int], y_pred: List[int], indices: List[int]) -> Dict[str, float]:
    if not indices:
        return {}
    y_t = [y_true[i] for i in indices]
    y_p = [y_pred[i] for i in indices]
    return compute_all_metrics(y_t, y_p)
