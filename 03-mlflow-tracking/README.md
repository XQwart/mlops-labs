# 03 · MLflow experiment tracking

The notebook `mlflow_lab3.ipynb` trains SGD, Ridge and Lasso regressors on UCI Student Performance and logs them to MLflow. It then picks the run with the best R² and serves it. The best one is Lasso: RMSE 2.02, R² 0.79.

```bash
mlflow ui
mlflow models serve -m runs:/<run_id>/model -p 5001 --env-manager local
```
