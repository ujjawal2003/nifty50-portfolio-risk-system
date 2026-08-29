import pandas as pd
from typing import Optional


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data from filepath."""
    return pd.read_csv(filepath, parse_dates=["Date"])


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    return df.drop_duplicates()


def handle_missing_values(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """Handle missing values in dataframe."""
    if method == "ffill":
        return df.fillna(method="ffill")
    elif method == "drop":
        return df.dropna()
    return df


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw price data."""
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    return df
