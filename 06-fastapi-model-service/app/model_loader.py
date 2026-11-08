"""
Load the trained model artifact and apply the exact preprocessing used in
training (see train.py): the fitted OrdinalEncoder and StandardScaler are
stored in the same pickle as the Lasso model.
"""
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"


class ModelService:
    """Wraps the model together with its fitted preprocessors."""

    def __init__(self, path: Path = MODEL_PATH):
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
        except FileNotFoundError:
            logger.error(f"Model file not found: {path}. Run `python train.py` first.")
            raise
        self.model = bundle["model"]
        self.encoder = bundle["encoder"]
        self.scaler = bundle["scaler"]
        self.feature_order = bundle["feature_order"]
        self.cat_columns = bundle["cat_columns"]
        logger.info(f"Model loaded from {path}")

    def preprocess(self, raw: dict) -> np.ndarray:
        """Apply the training-time preprocessing to one raw record."""
        df = pd.DataFrame([raw])[self.feature_order]
        df[self.cat_columns] = self.encoder.transform(df[self.cat_columns])
        return self.scaler.transform(df)

    def predict(self, raw: dict) -> float:
        """Predict the final grade G3 (clipped to the 0-20 scale)."""
        y_pred = self.model.predict(self.preprocess(raw))[0]
        return float(np.clip(y_pred, 0.0, 20.0))


_service: ModelService | None = None


def get_model_service() -> ModelService:
    """Lazily create the model service."""
    global _service
    if _service is None:
        _service = ModelService()
    return _service
