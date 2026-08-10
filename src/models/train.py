"""Stage 3 of the pipeline: train several models and track every run in MLflow.

Satisfies requirement #2 (train and evaluate multiple models) and requirement
#4 (track experiments, parameters, metrics, and model artifacts).

Each model gets its own MLflow run inside one experiment, which is what makes
the runs comparable side by side in the UI. Every run records:

* the resolved hyperparameters and the seed
* accuracy / precision / recall / F1 / ROC-AUC on the validation fold
* raw confusion counts, so false negatives are visible as a number
* a confusion-matrix and ROC plot as artifacts
* the fitted sklearn Pipeline, with a signature, as a model artifact

The winner is chosen on ``train.primary_metric`` (recall by default) and
written to ``models/best_model.pkl`` for the evaluation stage.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import load_params, resolve
from src.data.preprocess import make_pipeline
from src.models.metrics import (
    compute_metrics,
    confusion_counts,
    save_confusion_matrix,
    save_roc_curve,
)

logger = logging.getLogger(__name__)

# Maps a key in params.yaml -> the estimator class it configures. Adding a
# fourth model means adding one entry here and one block in params.yaml.
MODEL_BUILDERS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}


def build_estimator(name: str, config: dict, seed: int):
    """Instantiate an estimator from its params.yaml block.

    The seed is injected here rather than written into params.yaml three times,
    so a single ``seed:`` change reseeds the whole project.

    Raises
    ------
    KeyError
        If ``name`` is not a known model.
    """
    if name not in MODEL_BUILDERS:
        raise KeyError(
            f"unknown model {name!r}. Known: {sorted(MODEL_BUILDERS)}"
        )
    return MODEL_BUILDERS[name](random_state=seed, **config)


def split_xy(df: pd.DataFrame, target: str, id_column: str):
    """Split a processed frame into features, target, and the id column.

    ``customer_id`` is removed from the feature matrix here. It stays available
    as the third return value so the retention report can name customers.
    """
    ids = df[id_column] if id_column in df.columns else None
    drop = [c for c in (target, id_column) if c in df.columns]
    return df.drop(columns=drop), df[target], ids


def train_one(
    name: str,
    params: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    figures_dir: Path,
) -> tuple[object, dict[str, float], str]:
    """Fit one model, score it on the validation fold, and log an MLflow run.

    Returns the fitted pipeline, its validation scores, and the MLflow run id
    (needed later to promote the winner into the Model Registry).
    """
    target = params["data"]["target"]
    id_column = params["data"]["id_column"]
    seed = params["seed"]
    config = dict(params["train"]["models"][name])

    X_train, y_train, _ = split_xy(train_df, target, id_column)
    X_val, y_val, _ = split_xy(val_df, target, id_column)

    with mlflow.start_run(run_name=name):
        estimator = build_estimator(name, config, seed)
        pipeline = make_pipeline(params, estimator)

        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]

        scores = compute_metrics(y_val, y_pred, y_proba)
        counts = confusion_counts(y_val, y_pred)

        # --- parameters ---------------------------------------------------
        mlflow.log_param("model_type", name)
        mlflow.log_param("seed", seed)
        mlflow.log_param("val_size", params["preprocess"]["val_size"])
        mlflow.log_param("train_rows", len(train_df))
        for key, value in config.items():
            mlflow.log_param(f"model__{key}", value)

        # --- metrics ------------------------------------------------------
        mlflow.log_metrics(scores)
        mlflow.log_metrics({k: float(v) for k, v in counts.items()})
        mlflow.log_metric("fit_seconds", fit_seconds)

        # --- artifacts ----------------------------------------------------
        cm_path = save_confusion_matrix(
            y_val, y_pred, figures_dir / f"{name}_confusion_matrix.png", name
        )
        roc_path = save_roc_curve(
            y_val, y_proba, figures_dir / f"{name}_roc_curve.png", name
        )
        mlflow.log_artifact(str(cm_path), artifact_path="figures")
        mlflow.log_artifact(str(roc_path), artifact_path="figures")

        # A signature records the expected input schema and output type, so the
        # logged model documents how to call it.
        sample = X_val.head(100)
        signature = infer_signature(sample, pipeline.predict(sample))
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=X_val.head(5),
        )

        logger.info(
            "%-24s recall=%.4f f1=%.4f roc_auc=%.4f  (%.1fs, %d false negatives)",
            name,
            scores["recall"],
            scores["f1"],
            scores.get("roc_auc", float("nan")),
            fit_seconds,
            counts["false_negatives"],
        )

    # model_info.model_uri is the MLflow 3 "logged model" address. Registering
    # from it avoids the indirection warning that runs:/<id>/model now emits.
    return (
        pipeline,
        {**scores, **counts, "fit_seconds": fit_seconds},
        model_info.model_uri,
    )


def main(params_path: str | None = None) -> None:
    """DVC stage entry point: train every configured model, keep the best."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = load_params(params_path)

    processed_dir = resolve(params["data"]["processed_dir"])
    figures_dir = resolve("reports/figures")
    models_dir = resolve("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")

    # --- MLflow setup -----------------------------------------------------
    tracking_uri = params["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())

    primary = params["train"]["primary_metric"]
    results: dict[str, dict] = {}
    best_name, best_pipeline, best_score, best_uri = None, None, -1.0, None

    for name in params["train"]["models"]:
        pipeline, scores, model_uri = train_one(
            name, params, train_df, val_df, figures_dir
        )
        results[name] = scores

        if scores[primary] > best_score:
            best_name, best_pipeline = name, pipeline
            best_score, best_uri = scores[primary], model_uri

    # --- promote the winner into the Model Registry ----------------------
    # Registry needs a database-backed store; params.yaml uses sqlite for
    # exactly this reason. Failure here must not lose a completed training run,
    # so it is reported rather than raised.
    registered_name = params["mlflow"].get("registered_model_name")
    if registered_name and best_uri:
        try:
            version = mlflow.register_model(model_uri=best_uri, name=registered_name)
            logger.info(
                "registered %s version %s (%s)",
                registered_name,
                version.version,
                best_name,
            )
        except Exception as exc:  # noqa: BLE001 - registry is a nice-to-have
            logger.warning("model registration skipped: %s", exc)

    # --- persist the winner ----------------------------------------------
    model_path = models_dir / "best_model.pkl"
    joblib.dump(best_pipeline, model_path)
    logger.info(
        "best model: %s (%s=%.4f) -> %s", best_name, primary, best_score, model_path
    )

    metrics_path = resolve("reports/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "primary_metric": primary,
                "best_model": best_name,
                "models": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("wrote %s", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn models.")
    parser.add_argument("--params", default=None, help="path to params.yaml")
    main(parser.parse_args().params)
