import os
import joblib
from sklearn.ensemble import RandomForestRegressor

from src.config.config import MODEL_PATH
from src.utils.logger import get_logger

logger = get_logger("TrainModel")


def train_model(X_train, y_train, model_path=None):
    if model_path is None:
        model_path = MODEL_PATH

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    logger.info("Training RandomForestRegressor")
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    logger.info(f"Model saved to {model_path}")

    return model
