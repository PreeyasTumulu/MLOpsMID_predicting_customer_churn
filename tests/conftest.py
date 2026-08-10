"""Shared fixtures.

Everything here is **synthetic**. No test in this suite reads ``data/raw`` or
``data/processed``, and that is the single most important property of the test
suite: those directories are DVC-tracked and deliberately absent from git, so
they do not exist on the GitHub Actions runner. A test that opened the real CSV
would pass locally and fail in CI, taking requirements #5 and #7 down together.

The only real project file the tests touch is ``params.yaml``, which *is*
committed and therefore present everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_params

RAW_COLUMNS = [
    "CustomerID",
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
    "Churn",
]


def make_frame(n: int = 200, seed: int = 0, learnable: bool = True) -> pd.DataFrame:
    """Build a processed-style frame (snake_case, ``customer_id`` retained).

    ``learnable=True`` ties churn to ``support_calls`` with noise, so a model
    fitted on it scores meaningfully above chance. That lets the model tests
    assert real behaviour instead of merely "it ran".
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "age": rng.integers(18, 66, n),
            "gender": rng.choice(["Male", "Female"], n),
            "tenure": rng.integers(1, 61, n),
            "usage_frequency": rng.integers(1, 31, n),
            "support_calls": rng.integers(0, 11, n),
            "payment_delay": rng.integers(0, 31, n),
            "subscription_type": rng.choice(["Basic", "Standard", "Premium"], n),
            "contract_length": rng.choice(["Monthly", "Quarterly", "Annual"], n),
            "total_spend": rng.integers(100, 1001, n),
            "last_interaction": rng.integers(1, 31, n),
        }
    )

    if learnable:
        signal = df["support_calls"] > 5
        noise = rng.random(n) < 0.15
        df["churn"] = (signal ^ noise).astype(int)
    else:
        df["churn"] = rng.integers(0, 2, n)

    return df


@pytest.fixture
def params() -> dict:
    """The real ``params.yaml`` -- committed to git, so present in CI."""
    return load_params()


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A processed-style frame: snake_case columns, no nulls, id retained."""
    return make_frame()


@pytest.fixture
def raw_df(clean_df: pd.DataFrame) -> pd.DataFrame:
    """A raw-style frame with the original Kaggle headers."""
    renamed = clean_df.rename(
        columns=dict(zip(clean_df.columns, RAW_COLUMNS, strict=True))
    )
    return renamed


@pytest.fixture
def dirty_df(clean_df: pd.DataFrame) -> pd.DataFrame:
    """``clean_df`` plus the defects the real training master contains.

    One wholly-blank row (the real file has exactly one) and one row whose
    target is null, plus padded categorical values.
    """
    blank = {c: np.nan for c in clean_df.columns}
    null_target = clean_df.iloc[0].to_dict() | {"churn": np.nan}

    dirty = pd.concat(
        [clean_df, pd.DataFrame([blank, null_target])],
        ignore_index=True,
    )
    dirty.loc[0, "gender"] = "  Female  "
    return dirty


@pytest.fixture
def rule_df() -> pd.DataFrame:
    """A frame with a planted deterministic rule: ``support_calls >= 8``.

    Mirrors the structure the audit found in the real training master, so the
    detector is tested against a known-correct answer.
    """
    df = make_frame(n=500, seed=7, learnable=False)
    df["churn"] = (df["support_calls"] >= 8).astype(int)
    return df
