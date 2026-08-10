"""Tests for cleaning, splitting, and feature transformation (requirement #5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    build_preprocessor,
    clean,
    coerce_dtypes,
    drop_identifier,
    drop_invalid_rows,
    make_pipeline,
    split_features_target,
    train_val_split,
)


# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------
def test_drop_identifier_removes_customer_id(clean_df: pd.DataFrame) -> None:
    assert "customer_id" not in drop_identifier(clean_df).columns


def test_drop_identifier_is_a_noop_when_absent(clean_df: pd.DataFrame) -> None:
    without = clean_df.drop(columns=["customer_id"])
    assert list(drop_identifier(without).columns) == list(without.columns)


def test_drop_invalid_rows_removes_blank_and_null_target(
    clean_df: pd.DataFrame, dirty_df: pd.DataFrame
) -> None:
    """dirty_df adds exactly two unusable rows to clean_df."""
    assert len(drop_invalid_rows(dirty_df)) == len(clean_df)


def test_drop_invalid_rows_resets_the_index(dirty_df: pd.DataFrame) -> None:
    cleaned = drop_invalid_rows(dirty_df)
    assert cleaned.index.equals(pd.RangeIndex(len(cleaned)))


def test_coerce_dtypes_casts_target_to_int(clean_df: pd.DataFrame) -> None:
    floated = clean_df.assign(churn=clean_df["churn"].astype(float))
    assert coerce_dtypes(floated)["churn"].dtype.kind == "i"


def test_coerce_dtypes_strips_categorical_whitespace() -> None:
    df = pd.DataFrame({"gender": ["  Female  ", "Male "], "churn": [1, 0]})
    assert list(coerce_dtypes(df)["gender"]) == ["Female", "Male"]


def test_clean_chains_every_step(dirty_df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    result = clean(dirty_df)
    assert "customer_id" not in result.columns
    assert len(result) == len(clean_df)
    assert result["churn"].dtype.kind == "i"
    assert result.notna().all().all()


def test_clean_can_retain_the_identifier(dirty_df: pd.DataFrame) -> None:
    """The processed files keep customer_id so the retention report can name
    customers; the transformer is what keeps it out of the model."""
    assert "customer_id" in clean(dirty_df, keep_identifier=True).columns


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------
def test_split_features_target_separates_y(clean_df: pd.DataFrame) -> None:
    X, y = split_features_target(clean_df)
    assert "churn" not in X.columns
    assert y.name == "churn"
    assert len(X) == len(y) == len(clean_df)


def test_split_features_target_raises_without_target(clean_df: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="churn"):
        split_features_target(clean_df.drop(columns=["churn"]))


def test_train_val_split_honours_val_size(clean_df: pd.DataFrame, params: dict) -> None:
    train, val = train_val_split(clean_df, params)
    expected_val = round(len(clean_df) * params["preprocess"]["val_size"])
    assert abs(len(val) - expected_val) <= 1
    assert len(train) + len(val) == len(clean_df)


def test_train_val_split_preserves_class_balance(
    clean_df: pd.DataFrame, params: dict
) -> None:
    """Stratification: both folds must mirror the overall churn rate."""
    train, val = train_val_split(clean_df, params)
    overall = clean_df["churn"].mean()
    assert train["churn"].mean() == pytest.approx(overall, abs=0.02)
    assert val["churn"].mean() == pytest.approx(overall, abs=0.02)


def test_train_val_split_is_reproducible(clean_df: pd.DataFrame, params: dict) -> None:
    """Requirement #8: the same seed must yield byte-identical folds."""
    first, _ = train_val_split(clean_df, params)
    second, _ = train_val_split(clean_df, params)
    pd.testing.assert_frame_equal(first, second)


def test_train_val_split_changes_with_the_seed(
    clean_df: pd.DataFrame, params: dict
) -> None:
    first, _ = train_val_split(clean_df, params)
    second, _ = train_val_split(clean_df, {**params, "seed": params["seed"] + 1})
    assert not first.index.equals(second.index)


# --------------------------------------------------------------------------
# feature transformation
# --------------------------------------------------------------------------
def test_preprocessor_is_returned_unfitted(params: dict) -> None:
    """Fitting before the split would leak validation statistics."""
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        build_preprocessor(params).transform(pd.DataFrame())


def test_preprocessor_expands_categoricals(clean_df: pd.DataFrame, params: dict) -> None:
    """7 numeric + 2 gender + 3 subscription + 1 ordinal = 13 features."""
    X, _ = split_features_target(clean_df)
    assert build_preprocessor(params).fit_transform(X).shape == (len(clean_df), 13)


def test_preprocessor_excludes_the_identifier(
    clean_df: pd.DataFrame, params: dict
) -> None:
    """Leakage guard.

    customer_id survives into the processed CSVs on purpose. This asserts the
    structural reason that is safe: the ColumnTransformer selects from an
    explicit allow-list, so the identifier cannot reach an estimator even when
    it is present in the input frame.
    """
    X, _ = split_features_target(clean_df)
    assert "customer_id" in X.columns

    features = build_preprocessor(params).fit(X).get_feature_names_out()

    assert "customer_id" not in features
    assert not any("customer_id" in f for f in features)


def test_ordinal_encoding_preserves_contract_ranking(params: dict) -> None:
    """Monthly < Quarterly < Annual must survive as an ordering."""
    order = params["preprocess"]["ordinal_features"]["contract_length"]
    frame = pd.DataFrame(
        {
            **{c: 0 for c in params["preprocess"]["numeric_features"]},
            "gender": "Female",
            "subscription_type": "Basic",
            "contract_length": order,
        }
    )

    transformer = build_preprocessor(params).fit(frame)
    encoded = pd.DataFrame(
        transformer.transform(frame), columns=transformer.get_feature_names_out()
    )["contract_length"]

    assert list(encoded) == [0.0, 1.0, 2.0]


def test_preprocessor_scales_numeric_features(
    clean_df: pd.DataFrame, params: dict
) -> None:
    """StandardScaler output should be ~zero-mean, unit-variance."""
    X, _ = split_features_target(clean_df)
    transformer = build_preprocessor(params).fit(X)
    out = pd.DataFrame(
        transformer.transform(X), columns=transformer.get_feature_names_out()
    )

    for column in params["preprocess"]["numeric_features"]:
        assert out[column].mean() == pytest.approx(0.0, abs=1e-9)
        assert out[column].std(ddof=0) == pytest.approx(1.0, abs=1e-9)


def test_preprocessor_handles_unseen_categories(
    clean_df: pd.DataFrame, params: dict
) -> None:
    """An unknown category at inference must not crash the pipeline."""
    X, _ = split_features_target(clean_df)
    transformer = build_preprocessor(params).fit(X)

    unseen = X.head(1).copy()
    unseen["subscription_type"] = "Platinum"
    unseen["contract_length"] = "Biennial"

    assert np.isfinite(transformer.transform(unseen)).all()


def test_make_pipeline_wraps_preprocessor_and_model(params: dict) -> None:
    from sklearn.linear_model import LogisticRegression

    pipeline = make_pipeline(params, LogisticRegression())
    assert list(pipeline.named_steps) == ["preprocessor", "model"]
