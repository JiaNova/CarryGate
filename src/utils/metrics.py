"""Binary classification metrics used by Stage B."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def auprc_score(y_true: np.ndarray, probability: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability, dtype=float)
    return float(average_precision_score(y_true, probability))


def auroc_score(y_true: np.ndarray, probability: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability, dtype=float)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probability))


def brier_score(y_true: np.ndarray, probability: np.ndarray) -> float:
    return float(brier_score_loss(np.asarray(y_true).astype(int), np.asarray(probability)))


def binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    return {
        "auprc": auprc_score(y_true, probability),
        "auroc": auroc_score(y_true, probability),
        "brier": brier_score(y_true, probability),
    }
