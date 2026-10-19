"""Canonical evaluation metrics for the fraud-detection project.

PR-AUC is defined as **Average Precision** using
``sklearn.metrics.average_precision_score``. The trapezoidal variant is kept
only for auditing historical results and must not be used for model selection
or delivery claims.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = [
    "compute_binary_metrics",
    "pr_auc",
    "pr_auc_trapezoidal",
]


def _as_1d_arrays(y_true: Any, y_score: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize labels and scores to 1-D NumPy arrays."""
    labels = np.asarray(y_true).ravel()
    scores = np.asarray(y_score, dtype=float).ravel()

    if labels.size != scores.size:
        raise ValueError(
            f"y_true and y_score must have the same length, got "
            f"{labels.size} and {scores.size}"
        )
    if labels.size == 0:
        raise ValueError("y_true and y_score must not be empty")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if np.unique(labels).size < 2:
        raise ValueError(
            "y_true must contain both classes (0 and 1) for PR-AUC and ROC-AUC"
        )
    if not np.isfinite(scores).all():
        raise ValueError("y_score must contain only finite values")

    return labels.astype(np.int8), scores


def pr_auc(y_true: Any, y_score: Any) -> float:
    """Return the canonical PR-AUC: Average Precision.

    This is the project-wide definition. Use it for model selection, reports,
    and comparisons. Do not mix it with the trapezoidal implementation.
    """
    labels, scores = _as_1d_arrays(y_true, y_score)
    return float(average_precision_score(labels, scores))


def pr_auc_trapezoidal(y_true: Any, y_score: Any) -> float:
    """Return the historical trapezoidal PR-AUC.

    This exists only to explain older Phase 5/6 results. It is not the
    canonical project metric and should not be used for decisions.
    """
    labels, scores = _as_1d_arrays(y_true, y_score)
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))


def compute_binary_metrics(
    y_true: Any,
    y_score: Any,
    threshold: float = 0.5,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    include_trapezoidal: bool = False,
) -> dict[str, Any]:
    """Compute threshold-independent and threshold-dependent metrics.

    Parameters
    ----------
    y_true:
        Binary labels (0/1).
    y_score:
        Fraud probabilities or any continuous scores.
    threshold:
        Decision threshold applied to ``y_score``.
    cost_fp, cost_fn:
        Optional business costs used for ``expected_cost``.
    include_trapezoidal:
        Include the historical trapezoidal PR-AUC for audit purposes only.

    Returns
    -------
    dict
        A JSON-serializable dictionary with canonical metric names.
    """
    labels, scores = _as_1d_arrays(y_true, y_score)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    predictions = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    result: dict[str, Any] = {
        "threshold": float(threshold),
        "n_samples": int(labels.size),
        "n_positive": int(labels.sum()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": pr_auc(labels, scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "cost_fp": float(cost_fp),
        "cost_fn": float(cost_fn),
        "expected_cost": float(cost_fp * fp + cost_fn * fn),
    }
    if include_trapezoidal:
        result["pr_auc_trapezoidal"] = pr_auc_trapezoidal(labels, scores)
    return result
