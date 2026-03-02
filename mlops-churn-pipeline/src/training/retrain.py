import subprocess
import sys
from pathlib import Path
import pandas as pd
import joblib
import mlflow
from sklearn.metrics import f1_score

from src.utils.logger import get_logger
from src.drift.detect_drift import detect_drift
from src.features.build_features import build_features

logger = get_logger("RETRAIN")

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "raw" / "customer_data.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

logger.info("🚀 Starting retraining pipeline")
logger.info(f"Using Python executable: {sys.executable}")

try:
    # -------------------------------------------------
    # Step 1: Generate fresh data
    # -------------------------------------------------
    subprocess.run(
        [sys.executable, "-m", "src.data.generate_data"],
        check=True
    )
    logger.info("✅ New dataset generated")

    # -------------------------------------------------
    # Step 2: Drift Detection
    # -------------------------------------------------
    if not DATA_PATH.exists():
        logger.warning("Training data not found. Skipping retraining.")
        sys.exit(0)

    df = pd.read_csv(DATA_PATH)

    # Simulate live data (for demo purpose)
    live_df = df.sample(frac=0.3, random_state=42)

    drifted = detect_drift(df, live_df)

    if not drifted:
        logger.info("🟢 No drift detected. Skipping retraining.")
        sys.exit(0)

    logger.warning(f"⚠ Drift detected in features: {drifted}")

    # -------------------------------------------------
    # Step 3: Backup Existing Model (if exists)
    # -------------------------------------------------
    old_model = None
    if MODEL_PATH.exists():
        old_model = joblib.load(MODEL_PATH)
        logger.info("Existing model loaded for comparison")

    # -------------------------------------------------
    # Step 4: Retrain Model
    # -------------------------------------------------
    subprocess.run(
        [sys.executable, "-m", "src.training.train"],
        check=True
    )
    logger.info("✅ Model retraining completed")

    # -------------------------------------------------
    # Step 5: Compare Performance
    # -------------------------------------------------
    new_model = joblib.load(MODEL_PATH)

    df = build_features(df)
    X = df.drop("churn", axis=1)
    y = df["churn"]

    if old_model:
        old_preds = old_model.predict(X)
        new_preds = new_model.predict(X)

        old_f1 = f1_score(y, old_preds)
        new_f1 = f1_score(y, new_preds)

        logger.info(f"Old Model F1: {old_f1:.4f}")
        logger.info(f"New Model F1: {new_f1:.4f}")

        if new_f1 < old_f1:
            logger.warning("🔁 New model worse than old. Restoring previous model.")
            joblib.dump(old_model, MODEL_PATH)
        else:
            logger.info("🏆 New model improved or equal. Keeping updated model.")
    else:
        logger.info("No previous model found. Using newly trained model.")

except Exception:
    logger.exception("❌ Retraining pipeline failed")
    raise