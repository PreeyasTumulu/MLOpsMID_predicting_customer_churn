"""Tests for model construction, metrics, and prediction (requirement #5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data.preprocess import make_pipeline
from src.models.metrics import (
    METRIC_NAMES,
    compute_metrics,
    confusion_counts,
    save_confusion_matrix,
    save_roc_curve,
)
from src.models.train import MODEL_BUILDERS, build_estimator, split_xy

# Hand-worked example: tp=2 fp=1 fn=1 tn=2
Y_TRUE = [0, 0, 1, 1, 1, 0]
Y_PRED = [0, 1, 1, 1, 0, 0]
Y_PROBA = [0.1, 0.6, 0.9, 0.8, 0.3, 0.2]


# --------------------------------------------------------------------------
# estimator construction
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(MODEL_BUILDERS))
def test_build_estimator_returns_the_right_class(name: str) -> None:
    estimator = build_estimator(name, {}, seed=42)
    assert isinstance(estimator, MODEL_BUILDERS[name])


def test_every_configured_model_is_buildable(params: dict) -> None:
    """params.yaml and the code registry must not drift apart."""
    for name, config in params["train"]["models"].items():
        assert name in MODEL_BUILDERS, f"{name} configured but not implemented"
        build_estimator(name, dict(config), params["seed"])


def test_build_estimator_rejects_an_unknown_model() -> None:
    with pytest.raises(KeyError, match="unknown model"):
        build_estimator("gradient_descent_by_hand", {}, seed=42)


def test_build_estimator_injects_the_seed() -> None:
    """Requirement #8: the seed comes from params.yaml, not from each model."""
    assert build_estimator("random_forest", {}, seed=1234).random_state == 1234


def test_build_estimator_applies_hyperparameters() -> None:
    estimator = build_estimator("random_forest", {"n_estimators": 7}, seed=42)
    assert estimator.n_estimators == 7


# --------------------------------------------------------------------------
# feature/target separation
# --------------------------------------------------------------------------
def test_split_xy_drops_target_and_identifier(clean_df: pd.DataFrame) -> None:
    X, y, ids = split_xy(clean_df, "churn", "customer_id")
    assert "churn" not in X.columns
    assert "customer_id" not in X.columns
    assert y.name == "churn"
    assert list(ids) == list(clean_df["customer_id"])


def test_split_xy_tolerates_a_missing_identifier(clean_df: pd.DataFrame) -> None:
    X, y, ids = split_xy(clean_df.drop(columns=["customer_id"]), "churn", "customer_id")
    assert ids is None
    assert len(X) == len(y)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def test_compute_metrics_matches_hand_computed_values() -> None:
    scores = compute_metrics(Y_TRUE, Y_PRED, Y_PROBA)
    assert scores["accuracy"] == pytest.approx(4 / 6)
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == pytest.approx(2 / 3)
    assert scores["f1"] == pytest.approx(2 / 3)


def test_compute_metrics_returns_every_expected_key() -> None:
    assert set(compute_metrics(Y_TRUE, Y_PRED, Y_PROBA)) == set(METRIC_NAMES)


def test_compute_metrics_omits_roc_auc_without_probabilities() -> None:
    """Absent rather than faked -- a fabricated AUC would be worse than none."""
    assert "roc_auc" not in compute_metrics(Y_TRUE, Y_PRED)


def test_compute_metrics_omits_roc_auc_for_a_single_class() -> None:
    assert "roc_auc" not in compute_metrics([1, 1, 1], [1, 1, 1], [0.9, 0.8, 0.7])


def test_compute_metrics_survives_a_degenerate_model() -> None:
    """A model predicting all-zeros scores 0 on precision, it does not raise."""
    scores = compute_metrics([0, 1, 1], [0, 0, 0])
    assert scores["precision"] == 0.0
    assert scores["recall"] == 0.0


def test_confusion_counts_matches_hand_computed_values() -> None:
    assert confusion_counts(Y_TRUE, Y_PRED) == {
        "true_negatives": 2,
        "false_positives": 1,
        "false_negatives": 1,
        "true_positives": 2,
    }


def test_confusion_counts_sum_to_the_sample_size(clean_df: pd.DataFrame) -> None:
    counts = confusion_counts(clean_df["churn"], clean_df["churn"])
    assert sum(counts.values()) == len(clean_df)


@pytest.mark.parametrize("saver", [save_confusion_matrix, save_roc_curve])
def test_plot_helpers_write_a_file(tmp_path, saver) -> None:
    values = Y_PROBA if saver is save_roc_curve else Y_PRED
    path = saver(Y_TRUE, values, tmp_path / "plot.png", "test")
    assert path.is_file() and path.stat().st_size > 0


# --------------------------------------------------------------------------
# end-to-end model behaviour
# --------------------------------------------------------------------------
def test_pipeline_predicts_only_valid_labels(
    clean_df: pd.DataFrame, params: dict
) -> None:
    X, y, _ = split_xy(clean_df, "churn", "customer_id")
    pipeline = make_pipeline(params, LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert set(np.unique(predictions)) <= {0, 1}
    assert len(predictions) == len(X)


def test_pipeline_probabilities_are_valid(clean_df: pd.DataFrame, params: dict) -> None:
    X, y, _ = split_xy(clean_df, "churn", "customer_id")
    pipeline = make_pipeline(params, LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)

    proba = pipeline.predict_proba(X)

    assert ((proba >= 0) & (proba <= 1)).all()
    assert proba.sum(axis=1) == pytest.approx(1.0)


def test_model_learns_better_than_chance(clean_df: pd.DataFrame, params: dict) -> None:
    """The fixture ties churn to support_calls, so a fitted model must beat 0.5.

    Guards against a pipeline that runs but has silently stopped learning --
    a mangled encoder, say, or a shuffled target.
    """
    X, y, _ = split_xy(clean_df, "churn", "customer_id")
    pipeline = make_pipeline(params, RandomForestClassifier(n_estimators=25, random_state=0))
    pipeline.fit(X, y)

    scores = compute_metrics(y, pipeline.predict(X), pipeline.predict_proba(X)[:, 1])

    assert scores["roc_auc"] > 0.5
    assert scores["recall"] > 0.5


def test_training_is_reproducible(clean_df: pd.DataFrame, params: dict) -> None:
    """Requirement #8: identical seed, identical predictions."""
    X, y, _ = split_xy(clean_df, "churn", "customer_id")

    def fit_and_predict():
        pipeline = make_pipeline(params, build_estimator("random_forest", {"n_estimators": 15}, seed=params["seed"]))
        return pipeline.fit(X, y).predict_proba(X)[:, 1]

    np.testing.assert_array_equal(fit_and_predict(), fit_and_predict())
