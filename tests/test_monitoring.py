"""Tests for the audit and evaluation reporting (requirement #5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import compare_to_validation, rank_by_risk
from src.monitoring.audit import (
    PSI_MAJOR_SHIFT,
    candidate_thresholds,
    classify_psi,
    compare_distributions,
    evaluate_rule,
    find_deterministic_rules,
    population_stability_index,
    retest_rules,
)


# --------------------------------------------------------------------------
# deterministic-rule detection
# --------------------------------------------------------------------------
def test_evaluate_rule_returns_support_and_purity(rule_df: pd.DataFrame) -> None:
    """rule_df is built so support_calls >= 8 implies churn exactly."""
    support, purity = evaluate_rule(rule_df, "support_calls", 8, ">=", "churn")
    assert support == (rule_df["support_calls"] >= 8).sum()
    assert purity == 1.0


def test_evaluate_rule_handles_an_empty_selection(rule_df: pd.DataFrame) -> None:
    support, purity = evaluate_rule(rule_df, "support_calls", 999, ">=", "churn")
    assert support == 0
    assert np.isnan(purity)


def test_find_deterministic_rules_recovers_the_planted_rule(
    rule_df: pd.DataFrame,
) -> None:
    rules = find_deterministic_rules(rule_df, "churn", ["support_calls", "age"])

    found = [r for r in rules if r["feature"] == "support_calls" and r["direction"] == ">="]
    assert found, f"planted rule not detected; got {rules}"
    assert found[0]["threshold"] == 8
    assert found[0]["churn_rate"] == 1.0


def test_find_deterministic_rules_reports_the_widest_threshold(
    rule_df: pd.DataFrame,
) -> None:
    """>=8, >=9 and >=10 are all pure; only the loosest is informative."""
    rules = find_deterministic_rules(rule_df, "churn", ["support_calls"])
    thresholds = [r["threshold"] for r in rules if r["direction"] == ">="]
    assert thresholds == [8]


def test_find_deterministic_rules_ignores_low_support(rule_df: pd.DataFrame) -> None:
    """A rule covering a sliver of the data is noise, not a generating rule."""
    assert find_deterministic_rules(
        rule_df, "churn", ["support_calls"], min_support_frac=0.99
    ) == []


def test_find_deterministic_rules_finds_nothing_in_random_data(
    clean_df: pd.DataFrame,
) -> None:
    """No perfect separator should be reported for noisy data."""
    noisy = clean_df.assign(churn=np.resize([0, 1], len(clean_df)))
    assert find_deterministic_rules(noisy, "churn", ["age", "tenure"]) == []


def test_find_deterministic_rules_skips_absent_features(rule_df: pd.DataFrame) -> None:
    find_deterministic_rules(rule_df, "churn", ["not_a_column"])  # must not raise


def test_retest_rules_flags_a_rule_that_does_not_generalise(
    rule_df: pd.DataFrame, clean_df: pd.DataFrame
) -> None:
    """The real finding: pure on training, meaningless on the holdout."""
    rules = find_deterministic_rules(rule_df, "churn", ["support_calls"])
    holdout = clean_df.assign(churn=np.resize([0, 1], len(clean_df)))

    retested = retest_rules(rules, holdout, "churn")

    assert retested[0]["holds_on_holdout"] is False
    assert retested[0]["holdout_churn_rate"] < 0.99


def test_retest_rules_confirms_a_rule_that_does_generalise(
    rule_df: pd.DataFrame,
) -> None:
    rules = find_deterministic_rules(rule_df, "churn", ["support_calls"])
    retested = retest_rules(rules, rule_df, "churn")
    assert retested[0]["holds_on_holdout"] is True


def test_candidate_thresholds_is_bounded() -> None:
    """Continuous features must not trigger an exhaustive scan."""
    wide = pd.Series(np.linspace(0, 1_000_000, 500_000))
    assert len(candidate_thresholds(wide, max_points=50)) <= 50


def test_candidate_thresholds_uses_exact_values_when_few() -> None:
    series = pd.Series([1, 2, 3, 3, 2])
    assert list(candidate_thresholds(series)) == [1, 2, 3]


# --------------------------------------------------------------------------
# population stability index
# --------------------------------------------------------------------------
def test_psi_is_zero_for_an_identical_distribution(clean_df: pd.DataFrame) -> None:
    assert population_stability_index(clean_df["age"], clean_df["age"]) < 1e-9


def test_psi_grows_with_the_size_of_the_shift(clean_df: pd.DataFrame) -> None:
    reference = clean_df["age"]
    small = population_stability_index(reference, reference + 2)
    large = population_stability_index(reference, reference + 25)
    assert 0 < small < large


def test_psi_flags_a_major_shift(clean_df: pd.DataFrame) -> None:
    reference = clean_df["support_calls"]
    shifted = reference + 6  # comparable to the real train-vs-holdout gap
    assert population_stability_index(reference, shifted) > PSI_MAJOR_SHIFT


def test_psi_handles_a_constant_feature() -> None:
    """Degenerate input must return 0.0 rather than dividing by zero."""
    constant = pd.Series([5] * 100)
    assert population_stability_index(constant, constant) == 0.0


@pytest.mark.parametrize(
    ("psi", "verdict"),
    [(0.0, "stable"), (0.09, "stable"), (0.10, "moderate_shift"), (0.24, "moderate_shift"), (0.25, "major_shift"), (1.5, "major_shift")],
)
def test_classify_psi_bands(psi: float, verdict: str) -> None:
    assert classify_psi(psi) == verdict


def test_compare_distributions_reports_every_feature(clean_df: pd.DataFrame) -> None:
    features = ["age", "tenure", "support_calls"]
    report = compare_distributions(clean_df, clean_df, features)

    assert set(report) == set(features)
    for entry in report.values():
        assert entry["verdict"] == "stable"
        assert entry["psi"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------
# retention reporting
# --------------------------------------------------------------------------
def test_rank_by_risk_orders_by_descending_probability() -> None:
    ids = pd.Series([10, 20, 30, 40])
    ranked = rank_by_risk(ids, np.array([0.1, 0.9, 0.5, 0.7]), top_n=4)

    assert list(ranked["customer_id"]) == [20, 40, 30, 10]
    assert ranked["churn_probability"].is_monotonic_decreasing


def test_rank_by_risk_numbers_priorities_from_one() -> None:
    ranked = rank_by_risk(pd.Series([1, 2, 3]), np.array([0.3, 0.2, 0.1]), top_n=3)
    assert list(ranked["priority"]) == [1, 2, 3]


def test_rank_by_risk_truncates_to_top_n() -> None:
    ids = pd.Series(range(100))
    ranked = rank_by_risk(ids, np.linspace(0, 1, 100), top_n=10)
    assert len(ranked) == 10
    assert ranked["churn_probability"].iloc[0] == pytest.approx(1.0)


def test_rank_by_risk_falls_back_when_ids_are_missing() -> None:
    ranked = rank_by_risk(None, np.array([0.2, 0.8]), top_n=2)
    assert list(ranked["customer_id"]) == [1, 0]


def test_compare_to_validation_computes_deltas() -> None:
    deltas = compare_to_validation(
        {"recall": 0.80, "f1": 0.75}, {"recall": 0.90, "f1": 0.78}
    )
    assert deltas == {"delta_recall": -0.1, "delta_f1": -0.03}


def test_compare_to_validation_is_empty_without_a_baseline() -> None:
    assert compare_to_validation({"recall": 0.8}, None) == {}


def test_compare_to_validation_ignores_unmatched_keys() -> None:
    deltas = compare_to_validation({"recall": 0.8, "f1": 0.7}, {"recall": 0.9})
    assert set(deltas) == {"delta_recall"}
