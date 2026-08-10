"""Stage 4: score the winning model on the held-out master and rank customers.

This stage carries three of the client's five business requirements:

* **Identify likely leavers** -- predictions on the holdout master.
* **Prioritise retention effort** -- a ranked ``high_risk_customers.csv``,
  sorted by churn probability, so the business team works the top of a list
  instead of an unordered set of flags.
* **Monitor performance over time** -- the holdout master is not a random
  split of the training data. Its churn rate is ~47% against ~57% in the
  training master, so scoring against it measures how the model holds up on a
  differently-distributed population. The gap between validation and holdout
  metrics is the drift signal, and it is logged to MLflow as its own run so
  the comparison is visible next to the training runs.
"""

from __future__ import annotations

import argparse
import json
import logging

import joblib
import mlflow
import pandas as pd

from src.config import load_params, resolve
from src.models.metrics import (
    compute_metrics,
    confusion_counts,
    save_confusion_matrix,
    save_roc_curve,
)
from src.models.train import split_xy

logger = logging.getLogger(__name__)


def rank_by_risk(
    ids: pd.Series | None,
    probabilities,
    top_n: int,
) -> pd.DataFrame:
    """Return the ``top_n`` customers most likely to churn, highest first.

    A probability ranking rather than a 0/1 flag is what makes the output
    actionable: a retention team with capacity for 500 calls needs to know
    *which* 500, not that 30,000 customers are "at risk".
    """
    frame = pd.DataFrame(
        {
            "customer_id": ids if ids is not None else range(len(probabilities)),
            "churn_probability": probabilities,
        }
    )
    ranked = frame.sort_values("churn_probability", ascending=False).head(top_n)
    ranked.insert(0, "priority", range(1, len(ranked) + 1))
    return ranked.reset_index(drop=True)


def compare_to_validation(
    holdout: dict[str, float],
    validation: dict[str, float] | None,
) -> dict[str, float]:
    """Metric-by-metric delta between holdout and validation performance.

    A large negative delta means the model degraded on the shifted population,
    which is exactly the condition a monitoring job should alert on.
    """
    if not validation:
        return {}
    return {
        f"delta_{name}": round(holdout[name] - validation[name], 6)
        for name in holdout
        if name in validation and isinstance(validation[name], (int, float))
    }


def main(params_path: str | None = None) -> None:
    """DVC stage entry point: evaluate the best model on the holdout master."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = load_params(params_path)

    target = params["data"]["target"]
    id_column = params["data"]["id_column"]
    top_n = params["evaluate"]["top_n_high_risk"]

    model_path = resolve("models/best_model.pkl")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"no trained model at {model_path}. Run `python -m src.models.train` "
            "(or `dvc repro`) first."
        )
    pipeline = joblib.load(model_path)

    test_df = pd.read_csv(resolve(params["data"]["processed_dir"]) / "test.csv")
    X_test, y_test, ids = split_xy(test_df, target, id_column)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    scores = compute_metrics(y_test, y_pred, y_proba)
    counts = confusion_counts(y_test, y_pred)

    # --- drift: how did we do versus the validation fold? -----------------
    train_metrics_path = resolve("reports/metrics.json")
    validation_scores = None
    best_name = "unknown"
    if train_metrics_path.is_file():
        recorded = json.loads(train_metrics_path.read_text(encoding="utf-8"))
        best_name = recorded.get("best_model", "unknown")
        validation_scores = recorded.get("models", {}).get(best_name)
    deltas = compare_to_validation(scores, validation_scores)

    # --- retention priority list ------------------------------------------
    ranked = rank_by_risk(ids, y_proba, top_n)
    ranked_path = resolve("reports/high_risk_customers.csv")
    ranked_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(ranked_path, index=False)

    figures_dir = resolve("reports/figures")
    cm_path = save_confusion_matrix(
        y_test, y_pred, figures_dir / "holdout_confusion_matrix.png", "holdout"
    )
    roc_path = save_roc_curve(
        y_test, y_proba, figures_dir / "holdout_roc_curve.png", "holdout"
    )

    # --- log as its own MLflow run ----------------------------------------
    tracking_uri = params["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"holdout_eval__{best_name}"):
        mlflow.log_param("evaluated_model", best_name)
        mlflow.log_param("holdout_rows", len(test_df))
        mlflow.log_param("holdout_churn_rate", round(float(y_test.mean()), 4))
        mlflow.log_metrics(scores)
        mlflow.log_metrics({k: float(v) for k, v in counts.items()})
        if deltas:
            mlflow.log_metrics(deltas)
        mlflow.log_artifact(str(cm_path), artifact_path="figures")
        mlflow.log_artifact(str(roc_path), artifact_path="figures")
        mlflow.log_artifact(str(ranked_path), artifact_path="reports")

    report = {
        "evaluated_model": best_name,
        "holdout_rows": len(test_df),
        "holdout_churn_rate": round(float(y_test.mean()), 4),
        "metrics": scores,
        "confusion": counts,
        "drift_vs_validation": deltas,
    }
    out_path = resolve("reports/test_metrics.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("holdout recall=%.4f f1=%.4f", scores["recall"], scores["f1"])
    if deltas:
        logger.info("drift vs validation: %s", deltas)
    logger.info("wrote %s and %s", out_path.name, ranked_path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the best churn model.")
    parser.add_argument("--params", default=None, help="path to params.yaml")
    main(parser.parse_args().params)
