"""Prediction helpers used by the Streamlit application."""

from __future__ import annotations

import joblib
import pandas as pd

from src.config import MODEL_PATH


def load_model_bundle(path=MODEL_PATH) -> dict:
    """Load the trained model bundle."""
    return joblib.load(path)


def predict_purchase(input_data: dict, bundle: dict) -> dict:
    """Predict purchase probability and class for one visitor session."""
    input_frame = pd.DataFrame([input_data], columns=bundle["feature_columns"])
    probability = float(bundle["model"].predict_proba(input_frame)[:, 1][0])
    threshold = float(bundle["threshold"])
    return {
        "purchase_probability": probability,
        "prediction": int(probability >= threshold),
        "threshold": threshold,
    }
