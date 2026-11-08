import os
import sys
import pandas as pd
import yaml
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.getcwd())
from src.loggers import get_logger


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(config):
    logger = get_logger("TRAIN")
    df = pd.read_csv(config["featurize"]["output_path"])

    target = "quality"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    test_path = config["train"]["test_path"]
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    pd.concat(
        [X_test.reset_index(drop=True), y_test.reset_index(drop=True)],
        axis=1,
    ).to_csv(test_path, index=False)
    logger.info(f"Train size: {X_train.shape}, test size: {X_test.shape}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])
    params = {
        "model__n_estimators": config["train"]["n_estimators"],
        "model__max_depth": config["train"]["max_depth"],
    }
    grid = GridSearchCV(
        pipe, params,
        cv=config["train"]["cv"],
        n_jobs=-1,
        scoring="r2",
    )
    grid.fit(X_train, y_train)
    logger.info(f"Best params: {grid.best_params_}")
    logger.info(f"Best CV R2: {grid.best_score_:.4f}")

    model_path = config["train"]["model_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(grid.best_estimator_, model_path)


if __name__ == "__main__":
    config = load_config("src/config.yaml")
    train(config)