"""
ORM-модели SQLAlchemy для хранения истории предсказаний.
"""
from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.sql import func

from app.database import Base


class PredictionLog(Base):
    """
    Таблица для логирования всех предсказаний модели.
    Каждый POST /predict добавляет сюда запись.
    """
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)

    g1 = Column(Integer, nullable=False)
    g2 = Column(Integer, nullable=False)
    studytime = Column(Integer, nullable=False)
    failures = Column(Integer, nullable=False)
    absences = Column(Integer, nullable=False)
    age = Column(Integer, nullable=False)
    sex = Column(String(1), nullable=False)

    predicted_g3 = Column(Float, nullable=False)
    rounded_g3 = Column(Integer, nullable=False)

    timestamp = Column(DateTime(timezone=True), server_default=func.now())
