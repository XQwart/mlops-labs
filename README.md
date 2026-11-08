# MLOps Labs

The ML lifecycle in six small projects, from orchestration and experiment tracking to reproducible pipelines and model serving. I built them in a university MLOps course (2026). Each folder is self-contained and has its own README.

[![CI](https://github.com/XQwart/mlops-labs/actions/workflows/ci.yml/badge.svg)](https://github.com/XQwart/mlops-labs/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Airflow](https://img.shields.io/badge/Airflow-017CEE?style=flat-square&logo=apacheairflow&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-13ADC7?style=flat-square&logo=dvc&logoColor=white)
![Jenkins](https://img.shields.io/badge/Jenkins-D24939?style=flat-square&logo=jenkins&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)

| # | Project | What it shows | Stack |
|---|---------|---------------|-------|
| 01 | [Airflow ML pipeline](01-airflow-ml-pipeline) | A five-task DAG: download, clean, train, validate, save artifacts | Airflow, scikit-learn |
| 02 | [Airflow custom hook](02-airflow-custom-hook) | A custom `BaseHook` for a paginated REST API with basic auth, and the DAG that uses it | Airflow 3, Flask, Docker Compose |
| 03 | [MLflow experiment tracking](03-mlflow-tracking) | Four model families tracked and compared, with the best one served over REST | MLflow, scikit-learn |
| 04 | [Jenkins CI pipeline](04-jenkins-ci-pipeline) | Jenkins stages that download data, train with MLflow logging, deploy with `mlflow models serve` and run a health check | Jenkins, MLflow |
| 05 | [DVC pipeline](05-dvc-pipeline) | A reproducible four-stage pipeline with versioned data and metrics | DVC, scikit-learn |
| 06 | [FastAPI model service](06-fastapi-model-service) | The model from lab 03 behind an API that logs predictions to PostgreSQL, with 17 tests and Docker Compose | FastAPI, SQLAlchemy, PostgreSQL, pytest |

## Datasets

- [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) (UCI): predict the final grade G3. Labs 01, 03 and 06.
- [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) (UCI): predict the quality score. Lab 05.
- Used cars in Moldova: predict the price. Labs 02 and 04.

## History

The labs started as separate repositories. Their commit histories were merged in here, so `git log -- <folder>` shows how each one was built.
