"""Standalone model evaluation script."""

from __future__ import annotations

import joblib
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE
from src.data_processing import load_data, split_features_target
from src.evaluation import evaluate_classifier


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    metrics = evaluate_classifier(bundle["model"], X_test, y_test, bundle["threshold"])
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
