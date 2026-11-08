# 05 · DVC pipeline

A reproducible training pipeline managed by DVC. It predicts the quality score of red wine from its physicochemical properties (UCI Wine Quality).

## Stages ([`dvc.yaml`](dvc.yaml))

```
clean ──> featurize ──> train ──> evaluate
```

| Stage | What it does |
|-------|--------------|
| `clean` | Downloads the raw data, removes duplicates and missing values, clips outliers to the 1st–99th percentiles |
| `featurize` | Adds `total_acidity`, `free_to_total_sulfur`, `sugar_alcohol_ratio` and `acid_ph_ratio` |
| `train` | `RandomForestRegressor` with `GridSearchCV` (4-fold) on an 80% split |
| `evaluate` | RMSE, MAE and R² on the held-out 20%, written to `metrics/metrics.json` |

All parameters (URLs, paths, quantiles, split, grid) are in [`src/config.yaml`](src/config.yaml). DVC reruns only the stages whose inputs changed.

## Results

| RMSE | MAE | R² |
|------|-----|----|
| 0.600 | 0.472 | 0.350 |

## Run

```bash
pip install dvc pandas scikit-learn pyyaml joblib
dvc repro          # run or update the pipeline
dvc metrics show   # print metrics
dvc dag            # show the stage graph
```
