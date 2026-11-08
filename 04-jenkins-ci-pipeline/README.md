# 04 · Jenkins CI pipeline

A Jenkins job that downloads data, trains a model, deploys it and checks that it responds. The stages are in [`jenkins_pipline.sh`](jenkins_pipline.sh), one block per build step.

| Stage | What happens |
|-------|--------------|
| 1. download | Creates a venv, installs `requirements.txt`, runs `download.py` (downloads the used-cars dataset, cleans outliers, ordinal-encodes categoricals) |
| 2. train | `train_model.py`: `SGDRegressor` + `GridSearchCV` over penalty, loss and alpha, with a power-transformed target. Parameters, metrics and the model go to MLflow, and the path to the best run is written to `best_model.txt` |
| 3. deploy | `mlflow models serve` on port 5003, run in the background (`BUILD_ID=dontKillMe` stops Jenkins from killing it) |
| 4. healthy | A `curl` request to `/invocations` with a sample feature vector |

## Run locally

```bash
pip install -r requirements.txt
python download.py && python train_model.py > best_model.txt
mlflow models serve -m "$(tail -1 best_model.txt)" -p 5003 --env-manager local
```
