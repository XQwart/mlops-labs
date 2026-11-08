"""
CRUD-операции для таблицы prediction_logs.
"""
from sqlalchemy.orm import Session

from app import models
from app.schemas import StudentFeatures


def create_prediction(
    db: Session,
    features: StudentFeatures,
    predicted_g3: float,
) -> models.PredictionLog:
    """Сохранить запись о предсказании в БД."""
    db_record = models.PredictionLog(
        g1=features.G1,
        g2=features.G2,
        studytime=features.studytime,
        failures=features.failures,
        absences=features.absences,
        age=features.age,
        sex=features.sex,
        predicted_g3=predicted_g3,
        rounded_g3=int(round(predicted_g3)),
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record


def get_predictions(db: Session, skip: int = 0, limit: int = 100):
    """Получить список последних предсказаний."""
    return (
        db.query(models.PredictionLog)
        .order_by(models.PredictionLog.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_prediction(db: Session, prediction_id: int):
    """Получить конкретное предсказание по ID."""
    return (
        db.query(models.PredictionLog)
        .filter(models.PredictionLog.id == prediction_id)
        .first()
    )


def count_predictions(db: Session) -> int:
    """Общее количество предсказаний в БД."""
    return db.query(models.PredictionLog).count()
