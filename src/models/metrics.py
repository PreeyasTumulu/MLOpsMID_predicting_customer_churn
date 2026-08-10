"""Metric computation and diagnostic plots, shared by training and evaluation.

Kept in its own module so that ``train.py`` and ``evaluate.py`` cannot drift
apart in how they score a model, and so the unit tests can check the metric
maths directly against hand-computed values.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Non-interactive backend: these run under DVC and inside CI, where no display
# exists. Must be set before pyplot is imported.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Order matters only for readability of the reports.
METRIC_NAMES: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
)


def compute_metrics(
    y_true,
    y_pred,
    y_proba=None,
) -> dict[str, float]:
    """Score a set of predictions.

    Parameters
    ----------
    y_true, y_pred
        Ground-truth and predicted binary labels.
    y_proba
        Positive-class probabilities. Required for ROC-AUC; when omitted, the
        ``roc_auc`` key is simply absent rather than faked.

    Notes
    -----
    ``zero_division=0`` keeps a degenerate model (one that predicts a single
    class) from raising instead of scoring 0, which is the honest reading.
    """
    scores: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None and len(np.unique(y_true)) > 1:
        scores["roc_auc"] = float(roc_auc_score(y_true, y_proba))

    return scores


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    """Return the confusion matrix as named counts.

    ``false_negatives`` is the number the business actually cares about: a
    churner the model failed to flag, i.e. a customer lost without a chance to
    intervene.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def save_confusion_matrix(y_true, y_pred, path: str | Path, title: str) -> Path:
    """Render a confusion matrix to ``path`` and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["retained", "churned"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_roc_curve(y_true, y_proba, path: str | Path, title: str) -> Path:
    """Render an ROC curve to ``path`` and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name=title)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
