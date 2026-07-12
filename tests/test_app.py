from contextlib import contextmanager
from fastapi.testclient import TestClient
from src.app import app


@contextmanager
def client_context():
    with TestClient(app) as c:
        yield c


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


def test_predict_endpoint():
    with TestClient(app) as client:
        payload = {"age": 52.0, "sex": 1.0, "cp": 3.0, "trestbps": 172.0, "chol": 199.0,
                   "fbs": 1.0, "restecg": 0.0, "thalach": 162.0, "exang": 0.0, "oldpeak": 0.5,
                   "slope": 1.0, "ca": 0.0, "thal": 7.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "heart_disease_present" in data
        assert "confidence" in data
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0


def test_predict_invalid_input():
    with TestClient(app) as client:
        response = client.post("/predict", json={"age": "not_a_number"})
        assert response.status_code in (422, 500)


def test_predict_batch_consistency():
    payload = {"age": 52.0, "sex": 1.0, "cp": 3.0, "trestbps": 172.0, "chol": 199.0,
               "fbs": 1.0, "restecg": 0.0, "thalach": 162.0, "exang": 0.0, "oldpeak": 0.5,
               "slope": 1.0, "ca": 0.0, "thal": 7.0}
    with TestClient(app) as client:
        r1 = client.post("/predict", json=payload)
        r2 = client.post("/predict", json=payload)
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json() == r2.json()
