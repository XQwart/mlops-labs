# 02 · Airflow custom hook

An Airflow 3 setup (CeleryExecutor, PostgreSQL, Redis) that pulls data from a REST API through a custom hook, so the DAGs don't have to deal with connection details, auth or pagination.

## Components

| Path | Role |
|------|------|
| `cars-api/` | Flask API over the used-cars dataset. HTTP basic auth, `GET /cars?offset=&limit=` returns paginated JSON with a `total` count |
| `dags/hooks.py` | `CarsHook(BaseHook)` reads host, port and credentials from the Airflow connection `carsapi`, keeps a `requests.Session`, and yields records page by page |
| `dags/01_python.py` | Baseline DAG: plain `requests` inside a `PythonOperator`, then price statistics by year |
| `dags/02_hook.py` | The same data through `CarsHook`, then cleaning: removes duplicates and NaNs, label-encodes categorical columns, and saves the mappings to `label_mappings.json` |
| `compose.yaml`, `compose.override.yaml` | The official Airflow compose file, plus the API service and the `carsapi` connection passed as `AIRFLOW_CONN_CARSAPI` |

## Run

```bash
echo "AIRFLOW_UID=$(id -u)" > .env
docker compose up -d --build
# Airflow UI: http://localhost:8080 (airflow / airflow), API: http://localhost:8081
```

Trigger `02_hook`. Its output goes to `data/custom_hook/` and `data/cleaned/`.
