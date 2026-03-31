from datetime import datetime
import json
import logging
import os

import pandas as pd

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from hooks import CarsHook


def _fetch_cars(conn_id: str, templates_dict: dict,
                batch_size: int = 1000, **_):
    logger = logging.getLogger(__name__)
    output_path = templates_dict["output_path"]

    logger.info("Fetching all cars from the API...")
    hook = CarsHook(conn_id=conn_id)

    cars = list(hook.get_cars(batch_size=batch_size))
    logger.info(f"Fetched {len(cars)} car records")

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(cars, f)

    logger.info(f"Saved cars to {output_path}")


def _clean_cars_data(templates_dict: dict, **_):
    logger = logging.getLogger(__name__)
    input_path = templates_dict["input_path"]
    output_path = templates_dict["output_path"]

    logger.info(f"Loading raw data from {input_path}")
    with open(input_path, "r") as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data)
    logger.info(f"Raw data shape: {df.shape}")

    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {before - len(df)} duplicates")

    before = len(df)
    df = df.dropna()
    logger.info(f"Removed {before - len(df)} rows with missing values")

    categorical_columns = df.select_dtypes(
        include=["object"]).columns.tolist()
    logger.info(
        f"Encoding categorical columns: {categorical_columns}")

    label_mappings = {}
    for col in categorical_columns:
        df[col], uniques = pd.factorize(df[col])
        label_mappings[col] = list(uniques)

    logger.info(f"Cleaned data shape: {df.shape}")

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    df.to_json(output_path, orient="records")
    logger.info(f"Saved cleaned data to {output_path}")


    mapping_path = os.path.join(
        output_dir, "label_mappings.json")
    with open(mapping_path, "w") as f:
        json.dump(label_mappings, f,
                  ensure_ascii=False, indent=2)
    logger.info(f"Saved label mappings to {mapping_path}")


with DAG(
    dag_id="02_hook",
    description="Fetches car data and cleans it.",
    start_date=datetime(2026, 2, 3),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
) as dag:

    fetch_task = PythonOperator(
        task_id="fetch_cars",
        python_callable=_fetch_cars,
        op_kwargs={"conn_id": "carsapi"},
        templates_dict={
            "output_path": "/data/custom_hook/cars.json",
        },
    )

    clean_task = PythonOperator(
        task_id="clean_cars_data",
        python_callable=_clean_cars_data,
        templates_dict={
            "input_path": "/data/custom_hook/cars.json",
            "output_path":
                "/data/cleaned/cars_cleaned.json",
        },
    )

    fetch_task >> clean_task
