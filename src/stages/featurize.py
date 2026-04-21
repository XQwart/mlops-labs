import os
import sys
import pandas as pd
import yaml

sys.path.append(os.getcwd())
from src.loggers import get_logger


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def featurize(config):
    logger = get_logger("FEATURIZE")
    df = pd.read_csv(config["clean"]["output_path"])

    df["total_acidity"] = (
        df["fixed acidity"] + df["volatile acidity"] + df["citric acid"]
    )
    df["free_to_total_sulfur"] = (
        df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1.0)
    )
    df["sugar_alcohol_ratio"] = (
        df["residual sugar"] / (df["alcohol"] + 1.0)
    )
    df["acid_ph_ratio"] = df["total_acidity"] / df["pH"]

    logger.info(f"Shape after features: {df.shape}")

    out_path = config["featurize"]["output_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    config = load_config("src/config.yaml")
    featurize(config)