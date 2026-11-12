"""
Minimal FastAPI app serving the saved sklearn pipeline.

Run:
  uvicorn src.api:app --reload --port 8000
"""

from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("artifacts/customer_churn_pipeline.joblib")
app = FastAPI(title="Churn Prediction API (sklearn pipeline)")

pipeline = joblib.load(MODEL_PATH)


class CustomerInput(BaseModel):
    age: int = Field(..., ge=18, le=100)
    monthly_spend: float = Field(..., ge=0)
    tenure_months: int = Field(..., ge=0)
    plan_type: str
    region: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: CustomerInput):
    row = pd.DataFrame([payload.model_dump()])
    churn_probability = float(pipeline.predict_proba(row)[0, 1])
    prediction = int(churn_probability >= 0.5)

    return {
        "prediction": prediction,
        "churn_probability": round(churn_probability, 4),
        "threshold": 0.5,
    }