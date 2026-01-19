import os
import joblib
import pandas as pd

from src.feature_engineering import add_features
from src.config.config import INFERENCE_OUTPUT_PATH
from src.utils.logger import get_logger

logger = get_logger("InferencePipeline")


def run_inference_future_predict(
    model_path: str,        # trained model path
    input_path: str,        # future dataset CSV
    output_path: str = INFERENCE_OUTPUT_PATH
):
    """
    Run inference on a future dataset using a trained model.

    Steps:
    1. Load the future dataset
    2. Apply same feature engineering as training
    3. Encode categorical features using saved LabelEncoders
    4. Prepare features for prediction
    5. Load the model and generate predictions
    6. Save the predictions
    """

    logger.info("Starting inference pipeline")

    # Load future dataset
    df_future = pd.read_csv(input_path)
    df_future["date"] = pd.to_datetime(df_future["date"])
    df_future = df_future.dropna()

    logger.info(f"Loaded future data with shape: {df_future.shape}")

    # Apply SAME feature engineering
    df_future = add_features(df_future)
    logger.info("Feature engineering applied to future data")

    # Load label encoders
    store_le_path = "models/store_le.pkl"
    item_le_path = "models/item_le.pkl"

    if not (os.path.exists(store_le_path) and os.path.exists(item_le_path)):
        raise FileNotFoundError(
            "Label encoders not found. Expected store_le.pkl and item_le.pkl"
        )

    store_le = joblib.load(store_le_path)
    item_le = joblib.load(item_le_path)

    df_future["store_id"] = store_le.transform(df_future["store_id"])
    df_future["item_id"] = item_le.transform(df_future["item_id"])

    logger.info("Categorical features encoded")

    # Prepare features (drop date and target if present)
    cols_to_drop = ["date"]
    if "sales_qty" in df_future.columns:
        cols_to_drop.append("sales_qty")

    X_future = df_future.drop(columns=cols_to_drop)

    # Load trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")

    # Predict
    df_future["sales_qty_pred"] = model.predict(X_future)
    logger.info("Predictions generated")

    # Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_future.to_csv(output_path, index=False)

    logger.info(f"Inference results saved to {output_path}")
