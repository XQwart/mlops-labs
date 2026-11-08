"""
Train the Lasso model served by the API and save it together with its
preprocessors as a single artifact (model/model.pkl).

Reproduces the winning run from lab 03 (MLflow tracking): the same cleaning,
OrdinalEncoder + StandardScaler over the original column order, and a
GridSearchCV over Lasso. Saving the fitted encoder and scaler next to the
model guarantees the API preprocesses requests exactly like the training data.

Usage:
    python train.py
"""
import io
import pickle
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
MODEL_PATH = Path(__file__).parent / "model" / "model.pkl"
TARGET = "G3"


def load_dataset() -> pd.DataFrame:
    with urllib.request.urlopen(DATA_URL, timeout=60) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    with archive.open("student-mat.csv") as f:
        return pd.read_csv(f, sep=";")


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.drop_duplicates().dropna()
    low, high = data[TARGET].quantile([0.01, 0.99])
    data = data[(data[TARGET] >= low) & (data[TARGET] <= high)]
    return data.reset_index(drop=True)


def main() -> None:
    data = clean(load_dataset())
    X = data.drop(columns=[TARGET])
    y = data[TARGET].to_numpy()

    feature_order = X.columns.tolist()
    cat_columns = X.select_dtypes(include="object").columns.tolist()

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[cat_columns] = encoder.fit_transform(X[cat_columns])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    search = GridSearchCV(
        Lasso(random_state=42, max_iter=5000),
        {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0], "fit_intercept": [True, False]},
        cv=5,
        scoring="r2",
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    pred = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
    mae = float(mean_absolute_error(y_val, pred))
    r2 = float(r2_score(y_val, pred))
    print(f"Best params: {search.best_params_}")
    print(f"Validation: RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    bundle = {
        "model": model,
        "encoder": encoder,
        "scaler": scaler,
        "feature_order": feature_order,
        "cat_columns": cat_columns,
        "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
