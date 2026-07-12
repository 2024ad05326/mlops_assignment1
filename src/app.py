import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("models/best_model_pipeline.pkl")

class PatientFeatures(BaseModel):
    age: float; sex: float; cp: float; trestbps: float; chol: float
    fbs: float; restecg: float; thalach: float; exang: float; oldpeak: float
    slope: float; ca: float; thal: float

@app.post("/predict")
def predict(payload: PatientFeatures):
    input_df = pd.DataFrame([payload.model_dump()])
    
    # Extract prediction value safely
    prediction = int(model.predict(input_df)[0])
    
    # Calculate confidence percentage (probability of the predicted class)
    probabilities = model.predict_proba(input_df)[0]
    confidence = float(probabilities[prediction])
    
    return {
        "heart_disease_present": bool(prediction == 1),
        "confidence": confidence
    }

