# 01 · Airflow ML pipeline

An Apache Airflow DAG that trains a regression model on the UCI Student Performance dataset to predict a student's final grade (G3, on a 0–20 scale).

## DAG

```
download_data >> clear_data >> train_model >> validate_model >> save_artifacts
```

| Task | What it does |
|------|--------------|
| `download_data` | Downloads the UCI archive and extracts `student-mat.csv` |
| `clear_data` | Drops duplicates and missing values, encodes categorical features, removes outliers |
| `train_model` | Runs `SGDRegressor` with `GridSearchCV` over the penalty, alpha, l1_ratio, loss and intercept |
| `validate_model` | Scores the model on a hold-out split (RMSE, MAE, R²) |
| `save_artifacts` | Writes the model, encoder, scaler, metrics and a text report to `artifacts/` |

## Results

| RMSE | MAE | R² |
|------|-----|----|
| 2.017 | 1.312 | 0.790 |

Best parameters: `penalty=l1, alpha=0.1, l1_ratio=0.001, loss=squared_error`. See [`artifacts/metrics.json`](artifacts/metrics.json).

## Run

```bash
pip install -r requirements.txt
cp dags/ml_student_pipeline.py $AIRFLOW_HOME/dags/
airflow standalone   # then trigger the DAG from the UI
```
