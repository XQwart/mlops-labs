"""
FastAPI-приложение, оборачивающее модель Lasso (Лаба №3) в REST-сервис.
Эндпоинты:
    GET  /          — health check
    POST /predict   — получить предсказание G3 и сохранить в БД
    GET  /history   — список последних предсказаний
    GET  /history/{id} — конкретное предсказание
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
import uvicorn

from app import crud, models
from app.database import engine, get_db
from app.model_loader import get_model_service
from app.schemas import (
    StudentFeatures,
    PredictionResponse,
    PredictionRecord,
    HealthResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _init_db_with_retries(retries: int = 10, delay: float = 2.0) -> None:
    """Подождать пока БД поднимется (важно при docker-compose)."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            models.Base.metadata.create_all(bind=engine)
            logger.info("Таблицы БД готовы")
            return
        except OperationalError as e:
            last_err = e
            logger.warning(f"БД ещё не готова (попытка {attempt}/{retries}): {e}")
            time.sleep(delay)
    raise RuntimeError(f"Не удалось подключиться к БД: {last_err}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация на старте, очистка на остановке."""
    _init_db_with_retries()
    get_model_service()  # прогрев модели
    logger.info("Сервис готов к приёму запросов")
    yield


app = FastAPI(
    title="Student Performance Predictor",
    description="REST API для модели Lasso, обученной в Лабе №3 (MLflow).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_model=HealthResponse, summary="Health check")
def health_check(db: Session = Depends(get_db)):
    """Проверка состояния сервиса."""
    model_loaded = False
    db_connected = False
    try:
        get_model_service()
        model_loaded = True
    except Exception as e:
        logger.error(f"Модель не загружена: {e}")
    try:
        db.execute(models.PredictionLog.__table__.select().limit(1))
        db_connected = True
    except Exception as e:
        logger.error(f"БД недоступна: {e}")
    return HealthResponse(
        status="ok" if model_loaded and db_connected else "degraded",
        model_loaded=model_loaded,
        db_connected=db_connected,
    )


@app.post("/predict", response_model=PredictionResponse, summary="Predict G3")
def predict(features: StudentFeatures, db: Session = Depends(get_db)):
    """
    Предсказать итоговую оценку студента G3.
    Запрос и результат логируются в PostgreSQL.
    """
    try:
        service = get_model_service()
        predicted_g3 = service.predict(features.model_dump())
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    record = crud.create_prediction(db, features, predicted_g3)
    return PredictionResponse(
        id=record.id,
        predicted_g3=round(record.predicted_g3, 4),
        rounded_g3=record.rounded_g3,
        timestamp=record.timestamp,
    )


@app.get(
    "/history",
    response_model=list[PredictionRecord],
    summary="История предсказаний",
)
def history(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """Получить последние предсказания."""
    return crud.get_predictions(db, skip=skip, limit=limit)


@app.get(
    "/history/{prediction_id}",
    response_model=PredictionRecord,
    summary="Конкретное предсказание",
)
def history_item(prediction_id: int, db: Session = Depends(get_db)):
    record = crud.get_prediction(db, prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return record


@app.get("/stats", summary="Краткая статистика")
def stats(db: Session = Depends(get_db)):
    """Сколько всего предсказаний сделано."""
    return {"total_predictions": crud.count_predictions(db)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
