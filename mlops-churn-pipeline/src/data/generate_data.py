from faker import Faker
import random
import pandas as pd
import os

from src.utils.logger import get_logger
logger = get_logger("DATA")

fake = Faker()

def generate_data(n=200_000):
    logger.info("Generating synthetic dataset")
    records = []

    for _ in range(n):
        tenure = random.randint(1, 72)
        tickets = random.randint(0, 12)
        monthly = random.uniform(250, 1800)

        churn_prob = (
            0.5 if tenure < 6 else
            0.35 if tickets > 6 else
            0.15
        )

        churn = int(random.random() < churn_prob)

        records.append({
            "age": random.randint(18, 75),
            "gender": random.choice(["Male", "Female"]),
            "monthly_charges": round(monthly, 2),
            "tenure_months": tenure,
            "support_tickets": tickets,
            "payment_delay_days": random.randint(0, 20),
            "avg_usage_hours": round(random.uniform(0.2, 10), 2),
            "churn": churn
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_data()
    df.to_csv("data/raw/customer_data.csv", index=False)
    print("✅ Dataset generated")
    logger.info(f"Generated dataset with {len(df)} rows")   
    logger.info("Dataset saved to data/raw/customer_data.csv")
