# 01 · Airflow ML pipeline

An Airflow DAG that downloads UCI Student Performance, cleans it, tunes an `SGDRegressor` and saves the model to `artifacts/`. It predicts the final grade G3 (0–20). On the test split it scores RMSE 2.02, MAE 1.31 and R² 0.79.

```bash
pip install -r requirements.txt
cp dags/ml_student_pipeline.py $AIRFLOW_HOME/dags/
airflow standalone   # then trigger the DAG in the UI
```
