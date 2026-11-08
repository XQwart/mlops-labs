import os
import sys
import urllib.request
import pandas as pd
import yaml

sys.path.append(os.getcwd())
from src.loggers import get_logger


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_raw(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)


def clean(config):
    logger = get_logger("CLEAN")
    url = config["data"]["url"]
    raw_path = config["data"]["raw_path"]
    out_path = config["clean"]["output_path"]

    download_raw(url, raw_path)
    df = pd.read_csv(raw_path, sep=";")
    logger.info(f"Raw shape: {df.shape}")

    df = df.drop_duplicates().dropna().reset_index(drop=True)
    lo = config["clean"]["lower_quantile"]
    hi = config["clean"]["upper_quantile"]
    target = "quality"
    for col in df.columns:
        if col == target:
            continue
        q_lo = df[col].quantile(lo)
        q_hi = df[col].quantile(hi)
        df = df[(df[col] >= q_lo) & (df[col] <= q_hi)]

    df = df.reset_index(drop=True)
    logger.info(f"Clean shape: {df.shape}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    config = load_config("src/config.yaml")
    clean(config)