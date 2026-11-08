# 06 · FastAPI model service

The best model from [lab 03](../03-mlflow-tracking) (Lasso, R² 0.79) served behind a REST API. Every prediction is logged to PostgreSQL, the whole thing runs in Docker Compose, and pytest checks it.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check: model loaded, database reachable |
| `POST` | `/predict` | Predicts the final grade G3 from 32 student features. Inputs are validated by Pydantic |
| `GET` | `/history` | Logged predictions, paginated |
| `GET` | `/history/{id}` | One logged prediction |
| `GET` | `/stats` | Total number of logged predictions |

Interactive docs: `http://localhost:8000/docs`.

## Model artifact

`train.py` reproduces the lab 03 training run and saves **one** artifact, `model/model.pkl`, holding the model together with its fitted `OrdinalEncoder` and `StandardScaler`, the feature order and the validation metrics. The API applies exactly the training-time preprocessing.

```bash
python train.py   # Validation: RMSE=2.0176  MAE=1.3008  R2=0.7895
```

## Run

```bash
docker compose up -d --build    # API on :8000, PostgreSQL 16 in its own container
```

## Tests

```bash
pip install -r requirements.txt
pytest -q                       # 17 tests: endpoints, validation, DB logging, model
```

The tests run against an isolated SQLite database, and [CI](../.github/workflows/ci.yml) runs them on every push.
