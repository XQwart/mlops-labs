# MLOps Labs

[![CI](https://github.com/XQwart/mlops-labs/actions/workflows/ci.yml/badge.svg)](https://github.com/XQwart/mlops-labs/actions/workflows/ci.yml)

Six labs from a university MLOps course (2026). Each folder runs on its own.

| Lab | What it is |
|-----|------------|
| [01-airflow-ml-pipeline](01-airflow-ml-pipeline) | An Airflow DAG that trains a model to predict a student's final grade |
| [02-airflow-custom-hook](02-airflow-custom-hook) | A custom Airflow hook for a paginated REST API, in Docker Compose |
| [03-mlflow-tracking](03-mlflow-tracking) | Four regressors compared in MLflow, with the best one served over REST |
| [04-jenkins-ci-pipeline](04-jenkins-ci-pipeline) | A Jenkins job that downloads data, trains, deploys and checks the model |
| [05-dvc-pipeline](05-dvc-pipeline) | A reproducible DVC pipeline for wine quality |
| [06-fastapi-model-service](06-fastapi-model-service) | The lab 03 model behind FastAPI, with predictions logged to PostgreSQL |
