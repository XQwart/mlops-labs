# 02 · Airflow custom hook

`CarsHook` (`dags/hooks.py`) reads data from a paginated REST API with basic auth. The API is a small Flask service in `cars-api/`. The `02_hook` DAG loads the data through the hook and cleans it. `01_python` is the baseline: it calls the API with plain `requests` and computes price statistics by year.

```bash
echo "AIRFLOW_UID=$(id -u)" > .env
docker compose up -d --build
```

Airflow runs at http://localhost:8080 (login `airflow` / `airflow`) and the API at http://localhost:8081. The output goes to `data/`.
