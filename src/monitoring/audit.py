"""Data audit: deterministic-rule detection and distribution-shift measurement.

Why this module exists
----------------------
Training on the churn dataset produces ROC-AUC = 1.0000, which normally means
target leakage and normally means a broken pipeline. Here it means neither: the
training master contains a threshold rule that separates the target perfectly,
because that is how the data was generated.

A claim like that is worthless in a report unless something verifies it. This
module finds such rules automatically, re-tests each one on the holdout master
to show it does *not* generalise, and quantifies how far the two populations
have drifted apart. The output is written to ``reports/data_audit.json`` and
logged to MLflow, so the perfect score is accompanied by its own evidence.

The same two checks are what a production monitoring job would run on each new
batch of data: has a feature become suspiciously predictive, and has the input
distribution moved away from what the model was trained on.
"""

from __future__ import annotations

import argparse
import json
import logging

import mlflow
import numpy as np
import pandas as pd

from src.config import load_params, resolve

logger = logging.getLogger(__name__)

# A rule is only interesting if it covers a real share of the data; a rule that
# perfectly predicts nine customers is noise, not a generating rule.
DEFAULT_MIN_SUPPORT_FRAC = 0.01

# PSI convention used throughout the industry.
PSI_MINOR_SHIFT = 0.10
PSI_MAJOR_SHIFT = 0.25


def candidate_thresholds(series: pd.Series, max_points: int = 50) -> np.ndarray:
    """Threshold values to test for a feature.

    Uses every distinct value for low-cardinality integer features (so the
    exact cut point is found), and quantiles otherwise to bound the search.
    """
    unique = np.unique(series.dropna())
    if len(unique) <= max_points:
        return unique
    quantiles = np.linspace(0, 1, max_points)
    return np.unique(np.quantile(unique, quantiles))


def evaluate_rule(
    df: pd.DataFrame,
    feature: str,
    threshold: float,
    direction: str,
    target: str,
) -> tuple[int, float]:
    """Return ``(support, purity)`` for a single threshold rule.

    ``purity`` is the churn rate inside the selected subset: 1.0 means every
    matching customer churned.
    """
    mask = df[feature] >= threshold if direction == ">=" else df[feature] <= threshold
    subset = df.loc[mask, target]
    if subset.empty:
        return 0, float("nan")
    return len(subset), float(subset.mean())


def find_deterministic_rules(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    min_support_frac: float = DEFAULT_MIN_SUPPORT_FRAC,
    purity_threshold: float = 1.0,
) -> list[dict]:
    """Find single-feature threshold rules that (almost) perfectly imply churn.

    Scans ``feature >= t`` and ``feature <= t`` for each candidate threshold and
    keeps the rules whose churn rate meets ``purity_threshold`` while covering
    at least ``min_support_frac`` of the rows.

    Only the *widest* qualifying rule per feature and direction is kept --
    ``support_calls >= 6`` and ``>= 7`` and ``>= 8`` are the same finding, and
    the loosest threshold is the informative one.
    """
    min_support = int(len(df) * min_support_frac)
    found: list[dict] = []

    for feature in features:
        if feature not in df.columns:
            continue
        for direction in (">=", "<="):
            best: dict | None = None
            for threshold in candidate_thresholds(df[feature]):
                support, purity = evaluate_rule(
                    df, feature, threshold, direction, target
                )
                if support < min_support or not purity >= purity_threshold:
                    continue
                # Wider coverage = looser threshold = the rule worth reporting.
                if best is None or support > best["support"]:
                    best = {
                        "feature": feature,
                        "direction": direction,
                        "threshold": float(threshold),
                        "support": int(support),
                        "support_frac": round(support / len(df), 4),
                        "churn_rate": round(purity, 6),
                    }
            if best:
                found.append(best)

    return sorted(found, key=lambda r: r["support"], reverse=True)


def retest_rules(rules: list[dict], df: pd.DataFrame, target: str) -> list[dict]:
    """Re-evaluate discovered rules on a second dataset.

    A rule that holds at 1.00 on training and collapses on the holdout is the
    signature of two differently-generated populations rather than a real
    business relationship.
    """
    retested = []
    for rule in rules:
        support, purity = evaluate_rule(
            df, rule["feature"], rule["threshold"], rule["direction"], target
        )
        retested.append(
            {
                **rule,
                "holdout_support": int(support),
                "holdout_churn_rate": None if np.isnan(purity) else round(purity, 6),
                "holds_on_holdout": bool(not np.isnan(purity) and purity >= 0.99),
            }
        )
    return retested


def population_stability_index(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
) -> float:
    """Population Stability Index between a reference and a current sample.

    PSI < 0.10 is a stable feature, 0.10-0.25 a moderate shift, and > 0.25 a
    major shift that normally triggers retraining.

    Bin edges come from the *expected* distribution, since that is the one the
    model was trained on. A small epsilon keeps empty bins from producing
    infinities.
    """
    edges = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    eps = 1e-6
    expected_pct = np.histogram(expected.dropna(), bins=edges)[0] / len(expected)
    actual_pct = np.histogram(actual.dropna(), bins=edges)[0] / len(actual)
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def classify_psi(psi: float) -> str:
    """Turn a PSI number into the label a monitoring dashboard would show."""
    if psi >= PSI_MAJOR_SHIFT:
        return "major_shift"
    if psi >= PSI_MINOR_SHIFT:
        return "moderate_shift"
    return "stable"


def compare_distributions(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
) -> dict[str, dict]:
    """Per-feature PSI and means between the training and holdout populations."""
    report = {}
    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            continue
        psi = population_stability_index(reference[feature], current[feature])
        report[feature] = {
            "psi": round(psi, 4),
            "verdict": classify_psi(psi),
            "reference_mean": round(float(reference[feature].mean()), 4),
            "current_mean": round(float(current[feature].mean()), 4),
        }
    return report


def main(params_path: str | None = None) -> None:
    """DVC stage entry point: audit the processed data and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = load_params(params_path)

    target = params["data"]["target"]
    features = list(params["preprocess"]["numeric_features"])
    processed_dir = resolve(params["data"]["processed_dir"])

    train_df = pd.read_csv(processed_dir / "train.csv")
    holdout_df = pd.read_csv(processed_dir / "test.csv")

    # --- 1. deterministic rules ------------------------------------------
    rules = find_deterministic_rules(train_df, target, features)
    rules = retest_rules(rules, holdout_df, target)

    for rule in rules:
        logger.info(
            "RULE  %s %s %g -> churn %.4f on %d train rows (%.1f%%) | holdout %.4f %s",
            rule["feature"],
            rule["direction"],
            rule["threshold"],
            rule["churn_rate"],
            rule["support"],
            rule["support_frac"] * 100,
            rule["holdout_churn_rate"] or float("nan"),
            "HOLDS" if rule["holds_on_holdout"] else "DOES NOT HOLD",
        )
    if not rules:
        logger.info("no deterministic rules found above the support floor")

    # --- 2. distribution shift -------------------------------------------
    drift = compare_distributions(train_df, holdout_df, features)
    shifted = [f for f, d in drift.items() if d["verdict"] != "stable"]
    for feature in sorted(shifted, key=lambda f: -drift[f]["psi"]):
        entry = drift[feature]
        logger.info(
            "DRIFT %-18s psi=%.4f (%s)  mean %.2f -> %.2f",
            feature,
            entry["psi"],
            entry["verdict"],
            entry["reference_mean"],
            entry["current_mean"],
        )

    report = {
        "train_rows": len(train_df),
        "holdout_rows": len(holdout_df),
        "train_churn_rate": round(float(train_df[target].mean()), 4),
        "holdout_churn_rate": round(float(holdout_df[target].mean()), 4),
        "deterministic_rules": rules,
        "rules_that_generalise": sum(r["holds_on_holdout"] for r in rules),
        "distribution_shift": drift,
        "features_shifted": shifted,
        "interpretation": (
            "Perfect validation scores are explained by the deterministic "
            "rule(s) above, which hold on the training master but not on the "
            "holdout master. The two files were generated differently, so "
            "validation metrics overstate real-world performance."
        )
        if rules and not any(r["holds_on_holdout"] for r in rules)
        else "No unexplained perfect separation detected.",
    }

    out_path = resolve("reports/data_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote %s", out_path)

    # --- 3. log to MLflow so the audit sits beside the training runs ------
    tracking_uri = params["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="data_audit"):
        mlflow.log_param("deterministic_rules_found", len(rules))
        mlflow.log_param("rules_that_generalise", report["rules_that_generalise"])
        mlflow.log_param("features_shifted", ",".join(shifted) or "none")
        mlflow.log_metric("train_churn_rate", report["train_churn_rate"])
        mlflow.log_metric("holdout_churn_rate", report["holdout_churn_rate"])
        for feature, entry in drift.items():
            mlflow.log_metric(f"psi_{feature}", entry["psi"])
        mlflow.log_artifact(str(out_path), artifact_path="reports")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit data for leakage and drift.")
    parser.add_argument("--params", default=None, help="path to params.yaml")
    main(parser.parse_args().params)
