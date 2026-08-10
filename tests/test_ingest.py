"""Tests for data ingestion and schema validation (requirement #5)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.ingest import (
    EXPECTED_COLUMNS,
    SchemaError,
    describe,
    load_raw,
    normalise_column_name,
    normalise_columns,
    validate_schema,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Usage Frequency", "usage_frequency"),
        ("Support Calls", "support_calls"),
        ("Subscription Type", "subscription_type"),
        ("CustomerID", "customer_id"),  # camelCase boundary
        ("Age", "age"),
        ("  Churn  ", "churn"),  # surrounding whitespace
        ("Last-Interaction", "last_interaction"),  # hyphen
        ("already_snake", "already_snake"),  # idempotent
    ],
)
def test_normalise_column_name(raw: str, expected: str) -> None:
    assert normalise_column_name(raw) == expected


def test_normalise_columns_produces_the_expected_schema(raw_df: pd.DataFrame) -> None:
    assert set(normalise_columns(raw_df).columns) == set(EXPECTED_COLUMNS)


def test_normalise_columns_does_not_mutate_the_input(raw_df: pd.DataFrame) -> None:
    before = list(raw_df.columns)
    normalise_columns(raw_df)
    assert list(raw_df.columns) == before


def test_validate_schema_accepts_a_complete_frame(clean_df: pd.DataFrame) -> None:
    validate_schema(clean_df)  # must not raise


@pytest.mark.parametrize("missing", ["churn", "support_calls", "contract_length"])
def test_validate_schema_rejects_a_missing_column(
    clean_df: pd.DataFrame, missing: str
) -> None:
    with pytest.raises(SchemaError, match=missing):
        validate_schema(clean_df.drop(columns=[missing]))


def test_validate_schema_tolerates_extra_columns(clean_df: pd.DataFrame) -> None:
    """Extra columns are logged, not fatal -- the pipeline selects what it needs."""
    validate_schema(clean_df.assign(marketing_segment="A"))


def test_describe_reports_rows_nulls_and_churn_rate(clean_df: pd.DataFrame) -> None:
    summary = describe(clean_df, "train")
    assert summary["rows"] == len(clean_df)
    assert summary["null_cells"] == 0
    assert 0.0 <= summary["churn_rate"] <= 1.0


def test_describe_counts_nulls(dirty_df: pd.DataFrame) -> None:
    assert describe(dirty_df, "dirty")["null_cells"] > 0


def test_load_raw_error_mentions_dvc_pull(tmp_path) -> None:
    """A missing dataset is the most likely first-run failure for a new dev.

    The message has to point at `dvc pull` or requirement #8 stalls there.
    """
    with pytest.raises(FileNotFoundError, match="dvc pull"):
        load_raw(tmp_path / "does_not_exist.csv")


def test_load_raw_round_trips_a_csv(tmp_path, raw_df: pd.DataFrame) -> None:
    """Written with raw headers, read back normalised and validated."""
    path = tmp_path / "sample.csv"
    raw_df.to_csv(path, index=False)

    loaded = load_raw(path)

    assert set(loaded.columns) == set(EXPECTED_COLUMNS)
    assert len(loaded) == len(raw_df)
