"""Stage 1 of the pipeline: raw data ingestion and schema validation.

Reads the raw Kaggle CSVs, normalises the column names to snake_case, and
asserts the schema is what the rest of the pipeline expects.

Design note
-----------
Every function below takes and returns DataFrames rather than reading files
itself. That is deliberate: the unit tests exercise these functions with small
synthetic frames, because ``data/raw/`` is DVC-tracked and therefore *absent*
on the GitHub Actions runner. Any test that required the real CSV would fail
in CI and take requirements #5 and #7 down with it.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from src.config import load_params, resolve

logger = logging.getLogger(__name__)

# The schema the pipeline is written against, post-normalisation.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "customer_id",
    "age",
    "gender",
    "tenure",
    "usage_frequency",
    "support_calls",
    "payment_delay",
    "subscription_type",
    "contract_length",
    "total_spend",
    "last_interaction",
    "churn",
)

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


class SchemaError(ValueError):
    """Raised when an ingested frame does not match the expected schema."""


def normalise_column_name(name: str) -> str:
    """Convert a raw column header to snake_case.

    Handles both the spaced headers (``"Usage Frequency"``) and the camelCase
    identifier (``"CustomerID"``) present in the Kaggle files.

    >>> normalise_column_name("Usage Frequency")
    'usage_frequency'
    >>> normalise_column_name("CustomerID")
    'customer_id'
    """
    spaced = _CAMEL_BOUNDARY.sub("_", name.strip())
    return re.sub(r"[\s\-]+", "_", spaced).lower()


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with every column header normalised."""
    return df.rename(columns={c: normalise_column_name(c) for c in df.columns})


def validate_schema(
    df: pd.DataFrame,
    expected: tuple[str, ...] = EXPECTED_COLUMNS,
) -> None:
    """Assert that every expected column is present.

    Extra columns are tolerated and logged; missing ones are fatal, because
    silently training on a truncated schema is worse than failing loudly.

    Raises
    ------
    SchemaError
        If any expected column is absent.
    """
    actual = set(df.columns)
    missing = [c for c in expected if c not in actual]
    if missing:
        raise SchemaError(
            f"missing expected column(s): {missing}. Got: {sorted(actual)}"
        )

    unexpected = sorted(actual - set(expected))
    if unexpected:
        logger.warning("ignoring unexpected column(s): %s", unexpected)


def describe(df: pd.DataFrame, label: str) -> dict[str, object]:
    """Collect the handful of stats worth logging at ingest time."""
    summary: dict[str, object] = {
        "label": label,
        "rows": len(df),
        "columns": int(df.shape[1]),
        "null_cells": int(df.isna().sum().sum()),
    }
    if "churn" in df.columns:
        churn = df["churn"].dropna()
        summary["churn_rate"] = round(float(churn.mean()), 4) if len(churn) else None
    return summary


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read one raw CSV, normalise its headers, and validate the schema."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"raw data not found: {path}\n"
            "Run `dvc pull` to fetch the DVC-tracked datasets."
        )

    df = normalise_columns(pd.read_csv(path))
    validate_schema(df)
    return df


def main(params_path: str | None = None) -> None:
    """DVC stage entry point: ingest both raw files into ``data/interim``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = load_params(params_path)

    interim_dir = resolve(params["data"]["interim_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        source = resolve(params["data"][f"raw_{split}"])
        df = load_raw(source)
        logger.info("ingested %s -> %s", source.name, describe(df, split))

        destination = interim_dir / f"{split}.csv"
        df.to_csv(destination, index=False)
        logger.info("wrote %s (%d rows)", destination, len(df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest raw churn datasets.")
    parser.add_argument("--params", default=None, help="path to params.yaml")
    main(parser.parse_args().params)
