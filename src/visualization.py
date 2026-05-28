"""Plotting functions for EDA and model interpretation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree

from src.config import TARGET_COLUMN


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", palette="Set2")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 11


def run_eda(df: pd.DataFrame, output_dir) -> None:
    """Create and save beginner-friendly EDA charts."""
    set_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    sns.countplot(data=df, x=TARGET_COLUMN)
    plt.title("Purchase Distribution")
    plt.xlabel("Revenue")
    plt.ylabel("Session Count")
    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution.png", dpi=150)
    plt.close()

    missing_values = df.isna().sum().sort_values(ascending=False)
    plt.figure(figsize=(11, 5))
    sns.barplot(x=missing_values.index, y=missing_values.values)
    plt.title("Missing Values by Feature")
    plt.xlabel("Feature")
    plt.ylabel("Missing Values")
    plt.xticks(rotation=70, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "missing_values.png", dpi=150)
    plt.close()

    numeric_df = df.select_dtypes(include=["int64", "float64", "bool"]).copy()
    numeric_df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    plt.figure(figsize=(12, 9))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, linewidths=0.3)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()

    key_features = ["PageValues", "ExitRates", "BounceRates", "ProductRelated_Duration"]
    for feature in key_features:
        plt.figure()
        sns.histplot(data=df, x=feature, hue=TARGET_COLUMN, kde=True, bins=35)
        plt.title(f"{feature} Distribution by Purchase Outcome")
        plt.tight_layout()
        plt.savefig(output_dir / f"{feature.lower()}_distribution.png", dpi=150)
        plt.close()

    for feature in ["Month", "VisitorType", "Weekend"]:
        conversion = df.groupby(feature)[TARGET_COLUMN].mean().reset_index()
        plt.figure()
        sns.barplot(data=conversion, x=feature, y=TARGET_COLUMN)
        plt.title(f"Purchase Rate by {feature}")
        plt.xlabel(feature)
        plt.ylabel("Purchase Rate")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f"purchase_rate_by_{feature.lower()}.png", dpi=150)
        plt.close()


def plot_confusion_matrix(confusion_matrix_values, output_path) -> None:
    """Save the confusion matrix chart."""
    set_plot_style()
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix_values,
        display_labels=["No Purchase", "Purchase"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Pruned Decision Tree")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(feature_names, importances, output_path, top_n=20) -> pd.DataFrame:
    """Save and return the top feature importances."""
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False)
    top_features = importance_df.head(top_n)

    set_plot_style()
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_features, x="importance", y="feature", color="#4C78A8")
    plt.title("Top Decision Tree Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return importance_df


def plot_decision_tree_image(model, feature_names, output_path, max_depth=3) -> None:
    """Visualize the top levels of the trained tree."""
    plt.figure(figsize=(24, 12))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["No Purchase", "Purchase"],
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=8,
    )
    plt.title("Pruned Decision Tree Visualization")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
