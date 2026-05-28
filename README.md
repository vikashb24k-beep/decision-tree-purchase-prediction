# E-Commerce Purchase Prediction System

A complete end-to-end Machine Learning project that predicts whether an online visitor will make a purchase from browsing session behavior. The project uses a Decision Tree Classification model, includes EDA, preprocessing, pruning, evaluation, saved model artifacts, and a Streamlit web app.

## Project Features

- Exploratory Data Analysis with saved visual reports
- Missing value handling with scikit-learn imputers
- One-hot encoding for categorical variables
- Stratified train/test split for imbalanced purchase data
- Decision Tree baseline model and pruned Decision Tree model
- Pruning with `max_depth`, `min_samples_split`, `min_samples_leaf`, and `ccp_alpha`
- F1, accuracy, precision, recall, confusion matrix, and classification report
- Feature importance chart and decision tree visualization
- Streamlit app for real-time prediction and model insights
- Saved production-style model bundle using `joblib`

## Folder Structure

```text
E-Commerce-Purchase-Prediction/
├── data/
│   └── shop_smart_ecommerce.csv
├── notebooks/
├── src/
├── models/
├── outputs/
├── app/
│   └── app.py
├── requirements.txt
├── README.md
└── main.py
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Train the model and generate all outputs:

```bash
python main.py
```

Run standalone evaluation:

```bash
python -m src.evaluate_model
```

Launch the Streamlit app:

```bash
streamlit run app/app.py
```

## Model Performance

The training pipeline compares the unpruned tree with the pruned model. On the included dataset, the pruned model is designed to reach an F1 score above the requested 0.55 target.

Typical held-out test performance:

| Model | F1 Score | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|
| Unpruned Decision Tree | ~0.52 | ~0.85 | ~0.53 | ~0.50 |
| Pruned Decision Tree | ~0.62+ | ~0.84-0.88 | ~0.49-0.58 | ~0.76-0.85 |

Exact values are saved after training in:

```text
outputs/model_metrics.json
```

## Outputs

After running `python main.py`, the project creates:

- `models/decision_tree_purchase_model.joblib`
- `outputs/model_metrics.json`
- `outputs/test_predictions.csv`
- `outputs/feature_importance.csv`
- `outputs/feature_importance.png`
- `outputs/confusion_matrix.png`
- `outputs/decision_tree.png`
- EDA plots such as target distribution, missing values, correlation heatmap, and purchase-rate charts

## Technologies Used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- Streamlit

## Dataset Columns

The model uses session behavior features including `Administrative`, `ProductRelated`, `BounceRates`, `ExitRates`, `PageValues`, `Month`, `VisitorType`, `Weekend`, and other visitor/session attributes. The target column is `Revenue`.
