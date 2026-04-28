"""
Pydantic схемы для входных и выходных данных API.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime


class StudentFeatures(BaseModel):
    """
    Признаки студента из датасета Student Performance (UCI).
    Цель — предсказать итоговую оценку G3 (0-20).
    """
    # Демографические признаки
    sex: str = Field(..., description="Пол: 'M' или 'F'", examples=["M"])
    age: int = Field(..., ge=15, le=22, description="Возраст (15-22)", examples=[17])
    address: str = Field(..., description="Тип адреса: 'U' (городской) или 'R' (сельский)", examples=["U"])
    famsize: str = Field(..., description="Размер семьи: 'LE3' (≤3) или 'GT3' (>3)", examples=["GT3"])
    Pstatus: str = Field(..., description="Статус родителей: 'T' (вместе) или 'A' (раздельно)", examples=["T"])

    # Образование родителей
    Medu: int = Field(..., ge=0, le=4, description="Образование матери (0-4)", examples=[3])
    Fedu: int = Field(..., ge=0, le=4, description="Образование отца (0-4)", examples=[3])
    Mjob: str = Field(..., description="Работа матери", examples=["services"])
    Fjob: str = Field(..., description="Работа отца", examples=["teacher"])

    # Учебные признаки
    reason: str = Field(..., description="Причина выбора школы", examples=["course"])
    guardian: str = Field(..., description="Опекун", examples=["mother"])
    traveltime: int = Field(..., ge=1, le=4, description="Время в пути (1-4)", examples=[1])
    studytime: int = Field(..., ge=1, le=4, description="Время на учёбу (1-4)", examples=[2])
    failures: int = Field(..., ge=0, le=4, description="Кол-во неудач (0-4)", examples=[0])

    # Поддержка
    schoolsup: str = Field(..., description="Доп. поддержка школы: 'yes'/'no'", examples=["no"])
    famsup: str = Field(..., description="Поддержка семьи: 'yes'/'no'", examples=["yes"])
    paid: str = Field(..., description="Платные занятия: 'yes'/'no'", examples=["no"])
    activities: str = Field(..., description="Доп. активности: 'yes'/'no'", examples=["yes"])
    nursery: str = Field(..., description="Посещал детсад: 'yes'/'no'", examples=["yes"])
    higher: str = Field(..., description="Хочет ВО: 'yes'/'no'", examples=["yes"])
    internet: str = Field(..., description="Интернет дома: 'yes'/'no'", examples=["yes"])
    romantic: str = Field(..., description="В отношениях: 'yes'/'no'", examples=["no"])

    # Соц. признаки
    famrel: int = Field(..., ge=1, le=5, description="Отношения в семье (1-5)", examples=[4])
    freetime: int = Field(..., ge=1, le=5, description="Свободное время (1-5)", examples=[3])
    goout: int = Field(..., ge=1, le=5, description="Гуляет с друзьями (1-5)", examples=[3])
    Dalc: int = Field(..., ge=1, le=5, description="Алкоголь в будни (1-5)", examples=[1])
    Walc: int = Field(..., ge=1, le=5, description="Алкоголь в выходные (1-5)", examples=[2])
    health: int = Field(..., ge=1, le=5, description="Здоровье (1-5)", examples=[5])
    absences: int = Field(..., ge=0, le=100, description="Пропуски (0-100)", examples=[4])

    # Промежуточные оценки
    G1: int = Field(..., ge=0, le=20, description="Оценка за 1 период (0-20)", examples=[10])
    G2: int = Field(..., ge=0, le=20, description="Оценка за 2 период (0-20)", examples=[11])

    # школа (фиксированное поле, чаще всего не варьируется в API)
    school: str = Field(default="GP", description="Школа: 'GP' или 'MS'", examples=["GP"])


class PredictionResponse(BaseModel):
    """Ответ с предсказанием."""
    id: int
    predicted_g3: float = Field(..., description="Предсказанная итоговая оценка")
    rounded_g3: int = Field(..., description="Округленная итоговая оценка (0-20)")
    timestamp: datetime


class PredictionRecord(BaseModel):
    """Запись из истории предсказаний."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    predicted_g3: float
    rounded_g3: int
    g1: int
    g2: int
    studytime: int
    failures: int
    timestamp: datetime


class HealthResponse(BaseModel):
    """Ответ health-check."""
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    db_connected: bool
