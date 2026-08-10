"""Stage 2 of the pipeline: cleaning, splitting, and the feature transformer.

Two responsibilities, kept deliberately apart:

1. **Cleaning and splitting** happen here and are written to ``data/processed``.
2. **Feature transformation** is only *described* here, by
   :func:`build_preprocessor`. The transformer is fitted inside the training
   pipeline, on the training fold alone.

That separation is not stylistic. Fitting a ``StandardScaler`` before the
train/val split leaks validation statistics into training and quietly inflates
every metric you report. Composing the transformer into a ``Pipeline`` also
means the saved model carries its own preprocessing, so inference cannot drift
away from training.

As in ``ingest.py``, all functions operate on DataFrames so the tests can run
on synthetic data without ``data/raw/``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import load_params, resolve

logger = logging.getLogger(__name__)


def drop_identifier(df: pd.DataFrame, id_column: str = "customer_id") -> pd.DataFrame:
    """Remove the row-identifier column.

    ``customer_id`` carries no signal but is perfectly correlated with the row,
    so leaving it in is textbook target leakage for tree-based models.
    """
    if id_column not in df.columns:
        logger.debug("identifier %r already absent", id_column)
        return df.copy()
    return df.drop(columns=[id_column])


def drop_invalid_rows(df: pd.DataFrame, target: str = "churn") -> pd.DataFrame:
    """Drop rows that cannot be used for supervised training.

    The training master contains one wholly-blank row (12 null cells). That
    single row is also why every numeric column loads as float64 rather than
    int64 -- removing it is what makes :func:`coerce_dtypes` meaningful.
    """
    before = len(df)
    cleaned = df.dropna(how="all").dropna(subset=[target])
    dropped = before - len(cleaned)
    if dropped:
        logger.info("dropped %d row(s) with a null target or all-null", dropped)
    return cleaned.reset_index(drop=True)


def coerce_dtypes(df: pd.DataFrame, target: str = "churn") -> pd.DataFrame:
    """Cast the target to int and strip whitespace from categorical values.

    Must run *after* :func:`drop_invalid_rows`; casting a column that still
    contains NaN raises.
    """
    out = df.copy()
    if target in out.columns:
        out[target] = out[target].astype(int)

    for column in out.select_dtypes(include="object").columns:
        out[column] = out[column].str.strip()

    return out


def clean(df: pd.DataFrame, target: str = "churn", id_column: str = "customer_id"):
    """Run the full cleaning chain in the order the steps depend on."""
    return coerce_dtypes(
        drop_invalid_rows(drop_identifier(df, id_column), target),
        target,
    )


def split_features_target(df: pd.DataFrame, target: str = "churn"):
    """Separate the feature matrix from the target vector."""
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not present")
    return df.drop(columns=[target]), df[target]


def build_preprocessor(params: dict) -> ColumnTransformer:
    """Assemble the unfitted feature transformer.

    Returned unfitted on purpose -- see this module's docstring.

    - numeric      -> ``StandardScaler``
    - nominal      -> ``OneHotEncoder`` (unordered: gender, subscription type)
    - ordinal      -> ``OrdinalEncoder`` with an explicit Monthly < Quarterly <
      Annual ranking, which a one-hot would discard
    """
    cfg = params["preprocess"]
    numeric = list(cfg["numeric_features"])
    nominal = list(cfg["nominal_features"])
    ordinal_spec: dict[str, list[str]] = dict(cfg["ordinal_features"])

    ordinal_columns = list(ordinal_spec)
    ordinal_categories = [list(levels) for levels in ordinal_spec.values()]

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            (
                "nominal",
                # No `drop=` here: the small collinearity it would remove is
                # irrelevant to trees and absorbed by regularisation in the
                # linear model, and dropping interacts awkwardly with
                # handle_unknown="ignore".
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                nominal,
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(params: dict, estimator) -> Pipeline:
    """Compose the preprocessor and an estimator into one fittable object."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(params)),
            ("model", estimator),
        ]
    )


def train_val_split(df: pd.DataFrame, params: dict):
    """Stratified train/validation split, seeded from ``params.yaml``."""
    cfg = params["preprocess"]
    target = params["data"]["target"]

    stratify = df[target] if cfg.get("stratify", True) else None
    return train_test_split(
        df,
        test_size=cfg["val_size"],
        random_state=params["seed"],
        stratify=stratify,
    )


def main(params_path: str | None = None) -> None:
    """DVC stage entry point: clean, split, and write ``data/processed``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = load_params(params_path)

    interim_dir = resolve(params["data"]["interim_dir"])
    processed_dir = resolve(params["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    target = params["data"]["target"]
    id_column = params["data"]["id_column"]

    # --- training master -> train / validation ---------------------------
    train_master = clean(
        pd.read_csv(interim_dir / "train.csv"), target=target, id_column=id_column
    )

    sample_rows = params["train"].get("sample_rows")
    if sample_rows:
        train_master = train_master.sample(
            n=min(int(sample_rows), len(train_master)),
            random_state=params["seed"],
        ).reset_index(drop=True)
        logger.info("sampled down to %d rows for fast iteration", len(train_master))

    train_df, val_df = train_val_split(train_master, params)

    # --- testing master kept whole as a later-period holdout -------------
    # Its churn rate is ~47% against ~57% in the training master, so it is a
    # genuinely shifted distribution rather than a random split. That is what
    # the drift monitoring in stage 5 measures against.
    test_df = clean(
        pd.read_csv(interim_dir / "test.csv"), target=target, id_column=id_column
    )

    for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        destination: Path = processed_dir / f"{name}.csv"
        frame.to_csv(destination, index=False)
        logger.info(
            "wrote %s: %d rows, churn rate %.4f",
            destination.name,
            len(frame),
            frame[target].mean(),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and split churn data.")
    parser.add_argument("--params", default=None, help="path to params.yaml")
    main(parser.parse_args().params)
