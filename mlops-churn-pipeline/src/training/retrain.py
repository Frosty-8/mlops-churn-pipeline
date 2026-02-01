import subprocess
import sys
from src.utils.logger import get_logger

logger = get_logger("RETRAIN")

logger.info("Starting retraining pipeline")
logger.info(f"Using Python executable: {sys.executable}")

try:
    subprocess.run(
        [sys.executable, "-m", "src.data.generate_data"],
        check=True
    )
    logger.info("New dataset generated")

    subprocess.run(
        [sys.executable, "-m", "src.training.train"],
        check=True
    )
    logger.info("Model retraining completed successfully")

except Exception:
    logger.exception("Retraining pipeline failed")
    raise
