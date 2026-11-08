# 03 · MLflow experiment tracking

The notebook [`mlflow_lab3.ipynb`](mlflow_lab3.ipynb) trains four regressors on UCI Student Performance (target: final grade G3) and logs each one's parameters, metrics and model to MLflow. It then picks the best run and serves it over REST.

## Experiments

| Run | Model | Best params | RMSE | MAE | R² |
|-----|-------|-------------|------|-----|----|
| `SGD_GridSearch_basic` | SGDRegressor | alpha=0.05, l1_ratio=0.001 | 2.163 | 1.509 | 0.758 |
| `SGD_GridSearch_extended` | SGDRegressor | l1 penalty, epsilon-insensitive loss | 2.034 | 1.210 | 0.786 |
| `Ridge_GridSearch` | Ridge | alpha=10 | 2.162 | 1.501 | 0.758 |
| **`Lasso_GridSearch`** | **Lasso** | **alpha=0.1** | **2.018** | **1.301** | **0.790** |

The best run is chosen with `mlflow.search_runs` ordered by R², loaded back with `mlflow.sklearn.load_model`, and served:

```bash
mlflow ui                                   # browse the runs
mlflow models serve -m runs:/<run_id>/model -p 5001 --env-manager local
curl -X POST http://127.0.0.1:5001/invocations \
     -H "Content-Type: application/json" -d '{"inputs": [[...32 scaled features...]]}'
```

This model is served in production style in [06 · FastAPI model service](../06-fastapi-model-service).
