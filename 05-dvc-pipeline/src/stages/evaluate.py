import os
import sys
import json
import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

sys.path.append(os.getcwd())
from src.loggers import get_logger


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate(config):
    logger = get_logger("EVALUATE")
    df_test = pd.read_csv(config["train"]["test_path"])
    target = "quality"
    X_test = df_test.drop(columns=[target])
    y_test = df_test[target]

    model = joblib.load(config["train"]["model_path"])
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    logger.info(f"RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    out = config["evaluate"]["metrics_path"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    config = load_config("src/config.yaml")
    evaluate(config)