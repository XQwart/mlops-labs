# 06 · FastAPI model service

The lab 03 model (Lasso, R² 0.79) served with FastAPI. Every prediction is saved to PostgreSQL.

```bash
docker compose up -d --build
```

The API runs at http://localhost:8000 and its docs at `/docs`. `POST /predict` takes 32 student features and returns the predicted grade. `GET /history` lists past predictions.

`python train.py` retrains the model into `model/model.pkl`. Run the tests with `pytest`; they use SQLite and also run in CI.
