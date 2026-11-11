# 05 · DVC pipeline

Predicts red wine quality (UCI Wine Quality) with a `RandomForestRegressor`. The DVC pipeline has four stages: clean, featurize, train and evaluate. All parameters are in `src/config.yaml`, and the metrics are in `metrics/metrics.json` (RMSE 0.60, R² 0.35).

```bash
pip install dvc pandas scikit-learn pyyaml joblib
dvc repro
dvc metrics show
```
