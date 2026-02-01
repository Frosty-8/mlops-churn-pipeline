from fastapi import FastAPI
import joblib
import pandas as pd

from src.features.build_features import build_features
from src.utils.logger import get_logger

logger = get_logger("API")

app = FastAPI(title="Churn Prediction API")

try:
    model = joblib.load("models/best_model.pkl")
    logger.info("Model loaded successfully")
except Exception:
    logger.exception("Failed to load model")
    raise

@app.get("/")
def health():
    logger.info("Health check endpoint called")
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    logger.info(f"Prediction request received: {data}")

    try:
        df = pd.DataFrame([data])
        df = build_features(df)

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        logger.info(
            f"Prediction={int(prediction)} | Probability={probability:.4f}"
        )

        return {
            "churn_prediction": int(prediction),
            "churn_probability": round(float(probability), 4)
        }

    except Exception:
        logger.exception("Prediction failed")
        raise
