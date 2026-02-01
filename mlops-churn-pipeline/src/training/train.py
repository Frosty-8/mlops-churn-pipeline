import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.features.build_features import build_features
from src.utils.logger import get_logger

DATA_PATH = "data/raw/customer_data.csv"
MODEL_PATH = "models/best_model.pkl"
EXPERIMENT_NAME = "churn_prediction"

logger = get_logger("TRAINING")

def main():
    logger.info("🚀 Starting model training pipeline")

    # Load data
    logger.info(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Dataset loaded | Shape: {df.shape}")

    # Feature engineering
    logger.info("Building features")
    df = build_features(df)

    # Split features and target
    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info(f"Training samples: {X_train.shape[0]}")
    logger.info(f"Testing samples: {X_test.shape[0]}")

    # MLflow setup
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        logger.info("Training RandomForest model")

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)

        # Evaluation
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds)

        logger.info(f"Model training completed | F1-score: {f1:.4f}")

        # MLflow logging
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        logger.info(f"Model saved at: {MODEL_PATH}")

        print(f"✅ Model trained | F1-score: {f1:.4f}")


if __name__ == "__main__":
    main()
