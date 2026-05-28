"""Project configuration values."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
APP_DIR = BASE_DIR / "app"

DATA_PATH = DATA_DIR / "shop_smart_ecommerce.csv"
MODEL_PATH = MODEL_DIR / "decision_tree_purchase_model.joblib"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"

TARGET_COLUMN = "Revenue"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
