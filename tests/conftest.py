"""
Общие фикстуры pytest — изолированная in-memory SQLite-БД на каждый тест.
"""
import os
import sys
import pytest

# Используем SQLite до импорта приложения (чтобы не пытался подключиться к PG)
os.environ["DATABASE_URL"] = "sqlite:///./test_predictions.db"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from app.database import engine, SessionLocal
from app import models
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Создать таблицы один раз на сессию тестов."""
    models.Base.metadata.create_all(bind=engine)
    yield
    models.Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test_predictions.db"):
        os.remove("./test_predictions.db")


@pytest.fixture(autouse=True)
def clear_tables():
    """Очищать таблицы между тестами."""
    db = SessionLocal()
    db.query(models.PredictionLog).delete()
    db.commit()
    db.close()


@pytest.fixture
def client():
    """HTTP-клиент FastAPI."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_student():
    """Валидный пример студента для запросов."""
    return {
        "school": "GP",
        "sex": "M",
        "age": 17,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 3,
        "Fedu": 3,
        "Mjob": "services",
        "Fjob": "teacher",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 1,
        "studytime": 2,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 2,
        "health": 5,
        "absences": 4,
        "G1": 12,
        "G2": 13,
    }
