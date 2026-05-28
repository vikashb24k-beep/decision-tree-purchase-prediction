"""Streamlit app for real-time e-commerce purchase prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.config import DATA_PATH, MODEL_PATH, OUTPUT_DIR
from src.predict import predict_purchase


st.set_page_config(
    page_title="E-Commerce Purchase Prediction",
    page_icon="🛒",
    layout="wide",
)


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


def metric_card(label: str, value: str):
    st.metric(label=label, value=value)


def prediction_page(bundle, df):
    st.title("E-Commerce Purchase Prediction System")

    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Session Details")
        col1, col2, col3 = st.columns(3)
        administrative = col1.number_input("Administrative", min_value=0, value=2)
        informational = col2.number_input("Informational", min_value=0, value=0)
        product_related = col3.number_input("Product Related", min_value=0, value=12)

        col4, col5, col6 = st.columns(3)
        administrative_duration = col4.number_input("Administrative Duration", min_value=0.0, value=45.0)
        informational_duration = col5.number_input("Informational Duration", min_value=0.0, value=0.0)
        product_related_duration = col6.number_input("Product Related Duration", min_value=0.0, value=650.0)

        col7, col8, col9 = st.columns(3)
        bounce_rates = col7.slider("Bounce Rate", 0.0, 0.25, 0.02, 0.01)
        exit_rates = col8.slider("Exit Rate", 0.0, 0.25, 0.05, 0.01)
        page_values = col9.number_input("Page Values", min_value=0.0, value=10.0)

        col10, col11, col12 = st.columns(3)
        special_day = col10.slider("Special Day", 0.0, 1.0, 0.0, 0.2)
        month = col11.selectbox("Month", sorted(df["Month"].unique().tolist()), index=7)
        visitor_type = col12.selectbox("Visitor Type", sorted(df["VisitorType"].unique().tolist()))

        col13, col14, col15, col16 = st.columns(4)
        operating_systems = col13.selectbox("Operating System", sorted(df["OperatingSystems"].unique().tolist()))
        browser = col14.selectbox("Browser", sorted(df["Browser"].unique().tolist()))
        region = col15.selectbox("Region", sorted(df["Region"].unique().tolist()))
        traffic_type = col16.selectbox("Traffic Type", sorted(df["TrafficType"].unique().tolist()))
        weekend = st.toggle("Weekend Session", value=False)

        input_data = {
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_related_duration,
            "BounceRates": bounce_rates,
            "ExitRates": exit_rates,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_systems,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend,
        }

    with right:
        st.subheader("Prediction")
        result = predict_purchase(input_data, bundle)
        probability = result["purchase_probability"]
        label = "Purchase Likely" if result["prediction"] == 1 else "Purchase Unlikely"
        st.metric("Purchase Probability", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
        st.success(label) if result["prediction"] == 1 else st.info(label)

        metrics = bundle["metrics"]["pruned_decision_tree"]
        metric_cols = st.columns(2)
        metric_cols[0].metric("F1 Score", f"{metrics['f1_score']:.3f}")
        metric_cols[1].metric("Accuracy", f"{metrics['accuracy']:.3f}")
        metric_cols[0].metric("Precision", f"{metrics['precision']:.3f}")
        metric_cols[1].metric("Recall", f"{metrics['recall']:.3f}")

        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(x=["No Purchase", "Purchase"], y=[1 - probability, probability], ax=ax, palette="Set2")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Class Probability")
        st.pyplot(fig)


def insights_page(bundle, df):
    st.title("Model Insights")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sessions", f"{len(df):,}")
    col2.metric("Purchase Rate", f"{df['Revenue'].mean():.1%}")
    col3.metric("Decision Threshold", f"{bundle['threshold']:.2f}")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=df, x="Revenue", ax=ax, palette="Set2")
        ax.set_title("Purchase Distribution")
        st.pyplot(fig)

    with chart_col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        conversion = df.groupby("Month")["Revenue"].mean().reset_index()
        sns.barplot(data=conversion, x="Month", y="Revenue", ax=ax, palette="Set2")
        ax.set_title("Purchase Rate by Month")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    importance_path = OUTPUT_DIR / "feature_importance.csv"
    if importance_path.exists():
        importance_df = pd.read_csv(importance_path).head(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=importance_df, x="importance", y="feature", ax=ax, color="#4C78A8")
        ax.set_title("Top Feature Importances")
        st.pyplot(fig)


def performance_page(bundle):
    st.title("Model Performance")
    baseline = bundle["metrics"]["baseline_unpruned"]
    pruned = bundle["metrics"]["pruned_decision_tree"]

    comparison = pd.DataFrame(
        [
            {"Model": "Unpruned Decision Tree", "F1": baseline["f1_score"], "Accuracy": baseline["accuracy"], "Precision": baseline["precision"], "Recall": baseline["recall"]},
            {"Model": "Pruned Decision Tree", "F1": pruned["f1_score"], "Accuracy": pruned["accuracy"], "Precision": pruned["precision"], "Recall": pruned["recall"]},
        ]
    )
    st.dataframe(comparison, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    comparison_melted = comparison.melt(id_vars="Model", var_name="Metric", value_name="Score")
    sns.barplot(data=comparison_melted, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Before vs After Pruning")
    st.pyplot(fig)

    st.text(bundle["metrics"]["pruned_decision_tree"]["classification_report"])


def main():
    if not MODEL_PATH.exists():
        st.error("Trained model not found. Run `python main.py` from the project root.")
        st.stop()

    bundle = load_bundle()
    df = load_dataset()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "Insights", "Performance"])
    st.sidebar.caption("Decision Tree Classification")

    if page == "Prediction":
        prediction_page(bundle, df)
    elif page == "Insights":
        insights_page(bundle, df)
    else:
        performance_page(bundle)


if __name__ == "__main__":
    main()
