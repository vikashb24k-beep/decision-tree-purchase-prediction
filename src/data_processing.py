"""Data loading and preprocessing helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import TARGET_COLUMN


def load_data(path) -> pd.DataFrame:
    """Load the e-commerce sessions dataset."""
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame):
    """Return feature matrix and binary target vector."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def get_feature_groups(X: pd.DataFrame):
    """Detect numerical and categorical columns from the input dataframe."""
    categorical_features = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    numerical_features = [column for column in X.columns if column not in categorical_features]
    return numerical_features, categorical_features


def build_preprocessor(numerical_features, categorical_features) -> ColumnTransformer:
    """Create preprocessing steps for numeric and categorical features."""
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract feature names after preprocessing."""
    return preprocessor.get_feature_names_out().tolist()
