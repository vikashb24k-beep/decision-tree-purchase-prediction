"""Complete training pipeline for the purchase prediction model."""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    DATA_PATH,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    OUTPUT_DIR,
    PREDICTIONS_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    VALIDATION_SIZE,
)
from src.data_processing import (
    build_preprocessor,
    get_feature_groups,
    get_transformed_feature_names,
    load_data,
    split_features_target,
)
from src.evaluation import evaluate_classifier, find_best_threshold, save_metrics, save_predictions
from src.visualization import (
    plot_confusion_matrix,
    plot_decision_tree_image,
    plot_feature_importance,
    run_eda,
)


def build_decision_tree_pipeline(preprocessor, **tree_params) -> Pipeline:
    """Create a reproducible DecisionTreeClassifier pipeline."""
    model = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        **tree_params,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def tune_pruned_tree(X_train, y_train, numerical_features, categorical_features) -> GridSearchCV:
    """Tune all requested pruning controls using F1 as the scoring metric."""
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    pipeline = build_decision_tree_pipeline(preprocessor)

    parameter_grid = {
        "model__max_depth": [4, 5, 6, 7, 8],
        "model__min_samples_split": [10, 25, 50],
        "model__min_samples_leaf": [5, 10, 20],
        "model__ccp_alpha": [0.0, 0.0005, 0.001, 0.002],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=3,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search


def run_training_pipeline() -> dict:
    """Run EDA, train baseline and pruned models, evaluate, and save artifacts."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_PATH)
    run_eda(df, OUTPUT_DIR)

    X, y = split_features_target(df)
    numerical_features, categorical_features = get_feature_groups(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_model, X_validation, y_model, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    baseline_preprocessor = build_preprocessor(numerical_features, categorical_features)
    baseline_model = build_decision_tree_pipeline(baseline_preprocessor)
    baseline_model.fit(X_train, y_train)
    baseline_metrics = evaluate_classifier(baseline_model, X_test, y_test, threshold=0.50)

    grid_search = tune_pruned_tree(X_model, y_model, numerical_features, categorical_features)
    validation_probabilities = grid_search.best_estimator_.predict_proba(X_validation)[:, 1]
    best_threshold, validation_f1 = find_best_threshold(y_validation, validation_probabilities)

    best_params = {
        key.replace("model__", ""): value for key, value in grid_search.best_params_.items()
    }
    final_preprocessor = build_preprocessor(numerical_features, categorical_features)
    pruned_model = build_decision_tree_pipeline(final_preprocessor, **best_params)
    pruned_model.fit(X_train, y_train)
    pruned_metrics = evaluate_classifier(pruned_model, X_test, y_test, threshold=best_threshold)

    test_probabilities = pruned_model.predict_proba(X_test)[:, 1]
    save_predictions(X_test.index, y_test, test_probabilities, best_threshold, PREDICTIONS_PATH)

    preprocessor = pruned_model.named_steps["preprocessor"]
    tree_model = pruned_model.named_steps["model"]
    feature_names = get_transformed_feature_names(preprocessor)
    importance_df = plot_feature_importance(
        feature_names,
        tree_model.feature_importances_,
        OUTPUT_DIR / "feature_importance.png",
    )
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    plot_confusion_matrix(pruned_metrics["confusion_matrix"], OUTPUT_DIR / "confusion_matrix.png")
    plot_decision_tree_image(tree_model, feature_names, OUTPUT_DIR / "decision_tree.png")

    metrics = {
        "dataset_shape": list(df.shape),
        "target_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "baseline_unpruned": baseline_metrics,
        "pruned_decision_tree": pruned_metrics,
        "best_pruning_parameters": best_params,
        "validation_best_threshold": best_threshold,
        "validation_f1_at_best_threshold": validation_f1,
    }
    save_metrics(metrics, METRICS_PATH)

    model_bundle = {
        "model": pruned_model,
        "threshold": best_threshold,
        "feature_columns": X.columns.tolist(),
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "metrics": metrics,
        "best_pruning_parameters": best_params,
    }
    joblib.dump(model_bundle, MODEL_PATH)

    print("Training complete.")
    print(f"Baseline F1: {baseline_metrics['f1_score']:.3f}")
    print(f"Pruned Decision Tree F1: {pruned_metrics['f1_score']:.3f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    return metrics


if __name__ == "__main__":
    run_training_pipeline()
