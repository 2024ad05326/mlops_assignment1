from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_predict():
    payload = {"age": 52.0, "sex": 1.0, "cp": 3.0, "trestbps": 172.0, "chol": 199.0,
               "fbs": 1.0, "restecg": 0.0, "thalach": 162.0, "exang": 0.0, "oldpeak": 0.5,
               "slope": 1.0, "ca": 0.0, "thal": 7.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
