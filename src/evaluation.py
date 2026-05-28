"""Model evaluation utilities."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def find_best_threshold(y_true, probabilities, start=0.20, stop=0.80, step=0.01) -> tuple[float, float]:
    """Choose the probability threshold that maximizes F1 score."""
    thresholds = np.arange(start, stop + step, step)
    scores = [(threshold, f1_score(y_true, probabilities >= threshold)) for threshold in thresholds]
    best_threshold, best_score = max(scores, key=lambda item: item[1])
    return float(best_threshold), float(best_score)


def evaluate_classifier(model, X_test, y_test, threshold=0.50) -> dict:
    """Evaluate a classifier with probability thresholding."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["No Purchase", "Purchase"],
            zero_division=0,
        ),
    }


def save_metrics(metrics: dict, output_path) -> None:
    """Persist evaluation metrics as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def save_predictions(ids, y_true, probabilities, threshold, output_path) -> None:
    """Save test predictions for review and portfolio reporting."""
    predictions = (probabilities >= threshold).astype(int)
    prediction_frame = pd.DataFrame(
        {
            "row_id": ids,
            "actual_revenue": y_true.astype(int).values,
            "purchase_probability": probabilities,
            "predicted_revenue": predictions,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(output_path, index=False)
