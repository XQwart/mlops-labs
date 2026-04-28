"""
Автоматические тесты модели и API.
"""
import copy
import pytest


# ---------- 1. Health check ----------

def test_health_endpoint(client):
    """GET / должен вернуть статус и флаги модели/БД."""
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "degraded")
    assert body["model_loaded"] is True
    assert body["db_connected"] is True


# ---------- 2. /predict — валидные запросы ----------

def test_predict_valid(client, valid_student):
    """Корректные данные → корректный ответ модели."""
    resp = client.post("/predict", json=valid_student)
    assert resp.status_code == 200
    body = resp.json()
    assert "predicted_g3" in body
    assert "rounded_g3" in body
    assert "id" in body
    # Оценка должна быть в шкале 0-20
    assert 0.0 <= body["predicted_g3"] <= 20.0
    assert 0 <= body["rounded_g3"] <= 20


def test_predict_returns_float(client, valid_student):
    """predicted_g3 должен быть числом с плавающей точкой."""
    resp = client.post("/predict", json=valid_student)
    body = resp.json()
    assert isinstance(body["predicted_g3"], float)
    assert isinstance(body["rounded_g3"], int)


def test_predict_high_grades_yields_high_prediction(client, valid_student):
    """
    Студент с высокими G1, G2 должен получить большую оценку,
    чем студент с низкими G1, G2.
    """
    high = copy.deepcopy(valid_student)
    high["G1"] = 18
    high["G2"] = 19

    low = copy.deepcopy(valid_student)
    low["G1"] = 5
    low["G2"] = 6

    resp_h = client.post("/predict", json=high)
    resp_l = client.post("/predict", json=low)
    assert resp_h.status_code == 200 and resp_l.status_code == 200
    assert resp_h.json()["predicted_g3"] > resp_l.json()["predicted_g3"]


# ---------- 3. /predict — ошибки валидации ----------

def test_predict_missing_field(client, valid_student):
    """Если убрать обязательное поле — 422."""
    bad = copy.deepcopy(valid_student)
    bad.pop("G1")
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_invalid_age(client, valid_student):
    """Возраст вне диапазона → 422."""
    bad = copy.deepcopy(valid_student)
    bad["age"] = 99
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_invalid_grade(client, valid_student):
    """Оценка G1 > 20 → 422."""
    bad = copy.deepcopy(valid_student)
    bad["G1"] = 100
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


# ---------- 4. История предсказаний (БД) ----------

def test_history_empty_initially(client):
    """Сначала история пуста."""
    resp = client.get("/history")
    assert resp.status_code == 200
    assert resp.json() == []


def test_predict_writes_to_db(client, valid_student):
    """После /predict запись появляется в /history."""
    client.post("/predict", json=valid_student)
    resp = client.get("/history")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["g1"] == valid_student["G1"]
    assert rows[0]["g2"] == valid_student["G2"]


def test_history_item_by_id(client, valid_student):
    """GET /history/{id} возвращает конкретную запись."""
    pred = client.post("/predict", json=valid_student).json()
    rid = pred["id"]
    resp = client.get(f"/history/{rid}")
    assert resp.status_code == 200
    assert resp.json()["id"] == rid


def test_history_item_not_found(client):
    """GET /history/9999 → 404."""
    resp = client.get("/history/9999")
    assert resp.status_code == 404


def test_stats_increments(client, valid_student):
    """После 3-х предсказаний total_predictions == 3."""
    for _ in range(3):
        client.post("/predict", json=valid_student)
    resp = client.get("/stats")
    assert resp.status_code == 200
    assert resp.json()["total_predictions"] == 3


# ---------- 5. Прямые тесты модели (без HTTP) ----------

def test_model_loads():
    """Модель действительно загружается из артефакта."""
    from app.model_loader import get_model_service
    svc = get_model_service()
    assert svc.model is not None


def test_model_predict_directly(valid_student):
    """Можно вызвать модель напрямую, минуя HTTP."""
    from app.model_loader import get_model_service
    svc = get_model_service()
    pred = svc.predict(valid_student)
    assert isinstance(pred, float)
    assert 0.0 <= pred <= 20.0


@pytest.mark.parametrize("g1,g2", [(0, 0), (10, 10), (20, 20)])
def test_model_predict_grade_extremes(valid_student, g1, g2):
    """Модель выдаёт валидные значения на крайних точках шкалы."""
    from app.model_loader import get_model_service
    svc = get_model_service()
    data = copy.deepcopy(valid_student)
    data["G1"] = g1
    data["G2"] = g2
    pred = svc.predict(data)
    assert 0.0 <= pred <= 20.0
