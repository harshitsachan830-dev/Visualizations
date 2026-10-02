from __future__ import annotations

import time
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from lightgbm import LGBMClassifier, LGBMRegressor, record_evaluation
except ImportError as exc:
    raise RuntimeError("Install the API dependencies with: pip install -r requirements.txt") from exc

app = FastAPI(title="LightGBM Viz API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://lightgbm-viz.onrender.com",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class TrainingRequest(BaseModel):
    rows: list[dict[str, object]] = Field(min_length=4, max_length=100_000)
    target: str
    features: list[str] = Field(min_length=1, max_length=200)
    task_type: Literal["classification", "regression"] = "classification"
    surface_features: list[str] = Field(default_factory=list, max_length=2)
    model_variant: Literal["balanced", "shallow", "deep", "regularized"] = "balanced"
    include_details: bool = True


def serialize_tree(node: dict[str, object], feature_names: list[str]) -> dict[str, object]:
    if "split_index" not in node:
        return {
            "leaf": True,
            "value": float(node.get("leaf_value", 0)),
            "samples": int(node.get("leaf_count", 0)),
        }
    feature_index = int(node["split_feature"])
    return {
        "leaf": False,
        "feature": feature_names[feature_index],
        "threshold": float(node["threshold"]),
        "gain": float(node.get("split_gain", 0)),
        "samples": int(node.get("internal_count", 0)),
        "left": serialize_tree(node["left_child"], feature_names),
        "right": serialize_tree(node["right_child"], feature_names),
    }


def score_model(result: dict[str, object], task_type: str) -> float:
    return float(result["accuracy"] if task_type == "classification" else result["r2"])


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine": "LightGBM"}


@app.post("/api/train")
def train(request: TrainingRequest) -> dict[str, object]:
    frame = pd.DataFrame(request.rows)
    frame["_row_index"] = np.arange(len(frame))
    if request.target not in frame.columns:
        raise HTTPException(status_code=422, detail="The selected target column is missing.")

    features = [column for column in request.features if column in frame.columns and column != request.target]
    if not features:
        raise HTTPException(status_code=422, detail="Select at least one numeric feature other than the target.")

    for column in features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[request.target, *features])
    if len(frame) < 4:
        raise HTTPException(status_code=422, detail="At least four complete rows are required for training.")

    x_values = frame[features].astype(float)
    y_values = frame[request.target]
    if request.task_type == "classification":
        encoder = LabelEncoder()
        y_values = pd.Series(encoder.fit_transform(y_values.astype(str)), index=frame.index)
        class_counts = y_values.value_counts()
        if len(class_counts) < 2:
            raise HTTPException(status_code=422, detail="Classification requires at least two target classes.")
        if class_counts.min() < 2:
            raise HTTPException(status_code=422, detail="Each classification class needs at least two rows for a train/test split.")
        stratify = y_values
    else:
        y_values = pd.to_numeric(y_values, errors="coerce")
        valid = y_values.notna()
        x_values = x_values.loc[valid]
        y_values = y_values.loc[valid]
        if len(y_values) < 4:
            raise HTTPException(status_code=422, detail="Regression requires at least four numeric target values.")
        stratify = None

    row_indices = frame.loc[x_values.index, "_row_index"].astype(int).tolist()
    try:
        test_count = max(int(np.ceil(len(x_values) * 0.2)), len(np.unique(y_values)) if request.task_type == "classification" else 2)
        if test_count >= len(x_values):
            raise ValueError("Not enough rows for separate training and test sets.")
        x_train, x_test, y_train, y_test = train_test_split(
            x_values,
            y_values,
            test_size=test_count,
            random_state=42,
            stratify=stratify,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    history: dict[str, dict[str, list[float]]] = {}
    variants = {
        "balanced": {"n_estimators": 100, "learning_rate": 0.08, "num_leaves": 15, "max_depth": -1},
        "shallow": {"n_estimators": 80, "learning_rate": 0.08, "num_leaves": 7, "max_depth": 3},
        "deep": {"n_estimators": 120, "learning_rate": 0.06, "num_leaves": 31, "max_depth": -1},
        "regularized": {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 15, "max_depth": -1, "reg_lambda": 2.0},
    }
    common = {
        **variants[request.model_variant],
        "min_child_samples": max(2, min(20, len(x_train) // 10)),
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1,
    }
    if request.task_type == "classification":
        metric = "binary_logloss" if len(np.unique(y_train)) == 2 else "multi_logloss"
        estimator = LGBMClassifier(**common)
        fit_options = {"eval_metric": metric}
    else:
        estimator = LGBMRegressor(**common)
        fit_options = {"eval_metric": "rmse"}

    started = time.perf_counter()
    estimator.fit(
        x_train,
        y_train,
        eval_X=(x_train, x_test),
        eval_y=(y_train, y_test),
        eval_names=["train", "valid"],
        callbacks=[record_evaluation(history)],
        **fit_options,
    )
    elapsed = time.perf_counter() - started

    metric_key = next(iter(history.get("train", {})), None)
    train_curve = history.get("train", {}).get(metric_key, [])
    valid_curve = history.get("valid", {}).get(metric_key, [])
    chart_history = [
        {
            "iteration": index + 1,
            "train": float(train_curve[index]),
            "valid": float(valid_curve[index]),
        }
        for index in range(min(len(train_curve), len(valid_curve)))
        if index % 2 == 0 or index == len(train_curve) - 1
    ]

    result: dict[str, object] = {
        "model_variant": request.model_variant,
        "training_time": elapsed,
        "trees": int(estimator.booster_.current_iteration()),
        "history": chart_history,
        "feature_importance": [
            {"feature": feature, "value": float(value)}
            for feature, value in sorted(
                zip(features, estimator.booster_.feature_importance(importance_type="gain")),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
        "samples": int(len(frame)),
        "test_samples": int(len(y_test)),
    }
    if request.task_type == "classification":
        predicted_classes = encoder.inverse_transform(estimator.predict(x_values).astype(int))
        all_probabilities = estimator.predict_proba(x_values)
        result["predictions"] = [
            {
                "row_index": row_index,
                "prediction": str(predicted_class),
                "confidence": float(all_probabilities[index].max()),
            }
            for index, (row_index, predicted_class) in enumerate(zip(row_indices, predicted_classes))
        ]
        probabilities = estimator.predict_proba(x_test)
        result["accuracy"] = float(accuracy_score(y_test, estimator.predict(x_test)))
        result["log_loss"] = float(log_loss(y_test, probabilities, labels=estimator.classes_))
        predicted_codes = estimator.predict(x_values).astype(int)
    else:
        result["predictions"] = [
            {"row_index": row_index, "prediction": float(prediction)}
            for row_index, prediction in zip(row_indices, estimator.predict(x_values))
        ]
        predictions = estimator.predict(x_test)
        result["rmse"] = float(mean_squared_error(y_test, predictions) ** 0.5)
        result["r2"] = float(r2_score(y_test, predictions))

    if request.include_details:
        tree_info = estimator.booster_.dump_model()["tree_info"]
        result["tree"] = serialize_tree(tree_info[0]["tree_structure"], features)

        explain_positions = np.unique(np.linspace(0, len(x_values) - 1, min(len(x_values), 1_000), dtype=int))
        explain_rows = x_values.iloc[explain_positions]
        raw_contributions = np.asarray(estimator.booster_.predict(explain_rows, pred_contrib=True))
        if request.task_type == "classification":
            if len(estimator.classes_) == 2:
                contribution_matrix = raw_contributions.reshape(len(explain_rows), len(features) + 1)
                summary_values = np.abs(contribution_matrix[:, :-1]).mean(axis=0)
                row_contributions = contribution_matrix[:, :-1]
                result["shap_class"] = str(encoder.classes_[1])
            else:
                contribution_matrix = raw_contributions.reshape(len(explain_rows), len(estimator.classes_), len(features) + 1)
                summary_values = np.abs(contribution_matrix[:, :, :-1]).mean(axis=(0, 1))
                explained_codes = predicted_codes[explain_positions]
                row_contributions = contribution_matrix[np.arange(len(explain_rows)), explained_codes, :-1]
                result["shap_class"] = "predicted class per row"
        else:
            contribution_matrix = raw_contributions.reshape(len(explain_rows), len(features) + 1)
            summary_values = np.abs(contribution_matrix[:, :-1]).mean(axis=0)
            row_contributions = contribution_matrix[:, :-1]
            result["shap_class"] = "regression output"

        result["shap_summary"] = [
            {"feature": feature, "mean_abs_shap": float(value)}
            for feature, value in sorted(zip(features, summary_values), key=lambda item: item[1], reverse=True)
        ]
        shap_positions = np.unique(np.linspace(0, len(explain_rows) - 1, min(len(explain_rows), 120), dtype=int))
        result["shap_samples"] = [
            {
                "row_index": row_indices[explain_positions[position]],
                "values": [
                    {
                        "feature": feature,
                        "feature_value": float(explain_rows.iloc[position][feature]),
                        "shap_value": float(row_contributions[position, feature_index]),
                    }
                    for feature_index, feature in enumerate(features)
                ],
            }
            for position in shap_positions
        ]

        surface_features = [feature for feature in request.surface_features if feature in features][:2]
        if len(surface_features) < 2 and len(features) >= 2:
            surface_features = features[:2]
        if len(surface_features) == 2 and surface_features[0] != surface_features[1]:
            surface_x, surface_y = surface_features
            x_min, x_max = np.quantile(x_values[surface_x], [0.05, 0.95])
            y_min, y_max = np.quantile(x_values[surface_y], [0.05, 0.95])
            x_grid = np.linspace(x_min, x_max, 18)
            y_grid = np.linspace(y_min, y_max, 18)
            baseline = x_values.median()
            surface_rows = pd.DataFrame(
                [
                    {**baseline.to_dict(), surface_x: float(x_value), surface_y: float(y_value)}
                    for y_value in y_grid
                    for x_value in x_grid
                ],
                columns=features,
            )
            if request.task_type == "classification":
                surface_class = estimator.classes_[min(1, len(estimator.classes_) - 1)]
                z_grid = estimator.predict_proba(surface_rows)[:, int(surface_class)]
                surface_label = f"Probability: {encoder.inverse_transform([int(surface_class)])[0]}"
            else:
                z_grid = estimator.predict(surface_rows)
                surface_label = request.target
            result["surface"] = {
                "x_feature": surface_x,
                "y_feature": surface_y,
                "z_label": surface_label,
                "x_values": x_grid.tolist(),
                "y_values": y_grid.tolist(),
                "z_values": np.asarray(z_grid).reshape(len(y_grid), len(x_grid)).tolist(),
            }
        else:
            result["surface"] = None
    return result


@app.post("/api/compare")
def compare(request: TrainingRequest) -> dict[str, object]:
    variants = ["shallow", "balanced", "deep", "regularized"]
    results = []
    for variant in variants:
        fitted = train(request.model_copy(update={"model_variant": variant, "include_details": False}))
        results.append({
            "variant": variant,
            "accuracy": fitted.get("accuracy"),
            "log_loss": fitted.get("log_loss"),
            "r2": fitted.get("r2"),
            "rmse": fitted.get("rmse"),
            "training_time": fitted["training_time"],
            "trees": fitted["trees"],
        })
    best = max(results, key=lambda result: score_model(result, request.task_type))
    return {"models": results, "best_model": best["variant"], "selection_metric": "accuracy" if request.task_type == "classification" else "r2"}


@app.post("/api/tune")
def tune(request: TrainingRequest) -> dict[str, object]:
    comparison = compare(request)
    return {**comparison, "best_model": comparison["best_model"], "evaluated": len(comparison["models"])}
