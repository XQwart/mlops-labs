"""
Загрузка ML-модели из артефакта MLflow и применение того же препроцессинга,
что и при обучении в Лабе №3 (Lasso, datasetа Student Performance).
"""
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"

# Категориальные и числовые столбцы
CAT_COLUMNS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]
NUM_COLUMNS = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2",
]

FEATURE_ORDER = CAT_COLUMNS + NUM_COLUMNS


class ModelService:
    """Сервис, инкапсулирующий модель и препроцессоры."""

    def __init__(self):
        self.model = None
        self.ordinal_encoder = None
        self.scaler = None
        self._load_model()
        self._fit_preprocessors()

    def _load_model(self):
        """Загрузка модели Lasso, обученной в MLflow."""
        try:
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Модель успешно загружена из {MODEL_PATH}")
        except FileNotFoundError:
            logger.error(f"Файл модели не найден: {MODEL_PATH}")
            raise
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise

    def _fit_preprocessors(self):
        """
        Фит препроцессоров на синтетических данных, имитирующих
        диапазоны исходного датасета. В реальном проекте препроцессоры
        логируются вместе с моделью в MLflow.
        """
        # Возможные значения категориальных признаков (из описания UCI)
        sample = pd.DataFrame([
            {
                "school": "GP", "sex": "F", "address": "U", "famsize": "GT3",
                "Pstatus": "T", "Mjob": "at_home", "Fjob": "teacher",
                "reason": "course", "guardian": "mother",
                "schoolsup": "yes", "famsup": "no", "paid": "no",
                "activities": "no", "nursery": "yes", "higher": "yes",
                "internet": "no", "romantic": "no",
            },
            {
                "school": "MS", "sex": "M", "address": "R", "famsize": "LE3",
                "Pstatus": "A", "Mjob": "health", "Fjob": "services",
                "reason": "home", "guardian": "father",
                "schoolsup": "no", "famsup": "yes", "paid": "yes",
                "activities": "yes", "nursery": "no", "higher": "no",
                "internet": "yes", "romantic": "yes",
            },
            {
                "school": "GP", "sex": "F", "address": "U", "famsize": "GT3",
                "Pstatus": "T", "Mjob": "services", "Fjob": "other",
                "reason": "reputation", "guardian": "other",
                "schoolsup": "no", "famsup": "no", "paid": "no",
                "activities": "no", "nursery": "yes", "higher": "yes",
                "internet": "yes", "romantic": "no",
            },
            {
                "school": "GP", "sex": "M", "address": "U", "famsize": "GT3",
                "Pstatus": "T", "Mjob": "other", "Fjob": "at_home",
                "reason": "other", "guardian": "mother",
                "schoolsup": "no", "famsup": "yes", "paid": "no",
                "activities": "yes", "nursery": "yes", "higher": "yes",
                "internet": "yes", "romantic": "no",
            },
        ])
        self.ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self.ordinal_encoder.fit(sample[CAT_COLUMNS])

        rng = np.random.default_rng(42)
        n = 100
        num_sample = pd.DataFrame({
            "age": rng.integers(15, 23, n),
            "Medu": rng.integers(0, 5, n),
            "Fedu": rng.integers(0, 5, n),
            "traveltime": rng.integers(1, 5, n),
            "studytime": rng.integers(1, 5, n),
            "failures": rng.integers(0, 5, n),
            "famrel": rng.integers(1, 6, n),
            "freetime": rng.integers(1, 6, n),
            "goout": rng.integers(1, 6, n),
            "Dalc": rng.integers(1, 6, n),
            "Walc": rng.integers(1, 6, n),
            "health": rng.integers(1, 6, n),
            "absences": rng.integers(0, 30, n),
            "G1": rng.integers(0, 21, n),
            "G2": rng.integers(0, 21, n),
        })
        self.scaler = StandardScaler()
        self.scaler.fit(num_sample)
        logger.info("Препроцессоры подготовлены")

    def preprocess(self, raw: dict) -> pd.DataFrame:
        """Применить ту же предобработку, что и при обучении."""
        df = pd.DataFrame([raw])

        df[CAT_COLUMNS] = self.ordinal_encoder.transform(df[CAT_COLUMNS])
        df[NUM_COLUMNS] = self.scaler.transform(df[NUM_COLUMNS])
        return df[FEATURE_ORDER]

    def predict(self, raw: dict) -> float:
        """Предсказать G3 по сырым данным студента."""
        if self.model is None:
            raise RuntimeError("Модель не загружена")
        X = self.preprocess(raw)
        y_pred = self.model.predict(X.values)[0]
        return float(np.clip(y_pred, 0.0, 20.0))


_service: ModelService | None = None


def get_model_service() -> ModelService:
    """Lazy-инициализация сервиса модели."""
    global _service
    if _service is None:
        _service = ModelService()
    return _service
