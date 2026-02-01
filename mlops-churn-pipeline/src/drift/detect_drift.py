import pandas as pd
from scipy.stats import ks_2samp
from src.utils.logger import get_logger

logger = get_logger("DRIFT")

TRAIN_PATH = "data/raw/customer_data.csv"
LIVE_PATH = "data/live/live_data.csv"

def detect_drift(train_df, live_df):
    drifted_features = []

    for col in train_df.columns:
        if col == "churn":
            continue

        stat, p = ks_2samp(train_df[col], live_df[col])
        if p < 0.05:
            drifted_features.append(col)

    return drifted_features

if __name__ == "__main__":
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        live_df = pd.read_csv(LIVE_PATH)

        drifted = detect_drift(train_df, live_df)

        if drifted:
            logger.warning(f"Drift detected in features: {drifted}")
        else:
            logger.info("No drift detected")

    except FileNotFoundError:
        logger.info("Live data not available yet")
    except Exception:
        logger.exception("Drift detection failed")
