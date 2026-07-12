import logging
import time
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel
import src.monitoring as monitoring

logger = logging.getLogger(__name__)

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = joblib.load("models/best_model_pipeline.pkl")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    yield


app = FastAPI(title="Heart Disease Risk Predictor", lifespan=lifespan)


class PatientFeatures(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(request: Request, payload: PatientFeatures):
    start = time.time()
    try:
        input_df = pd.DataFrame([payload.model_dump()])
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(probabilities[prediction])

        monitoring.PREDICTION_COUNTER.labels(status='success', prediction=str(prediction)).inc()
        monitoring.PREDICTION_LATENCY.observe(time.time() - start)
        monitoring.CONFIDENCE_GAUGE.set(confidence)

        client_host = request.client.host if request.client else "unknown"
        logger.info(f"Prediction request from {client_host} -> prediction={prediction}, confidence={confidence:.4f}")

        return {
            "heart_disease_present": bool(prediction == 1),
            "confidence": confidence
        }
    except Exception as e:
        monitoring.PREDICTION_COUNTER.labels(status='error', prediction='unknown').inc()
        logger.error(f"Prediction error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
