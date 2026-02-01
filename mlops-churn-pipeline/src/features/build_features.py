import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["high_risk_user"] = (
        (df["support_tickets"] > 5) &
        (df["payment_delay_days"] > 5)
    ).astype(int)

    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins=[0, 6, 24, 72],
        labels=[0, 1, 2]
    ).astype(int)

    df["gender"] = df["gender"].map({"Male": 0, "Female": 1})

    return df
