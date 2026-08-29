import pandas as pd
import numpy as np


def validate_date_column(df: pd.DataFrame, date_col: str = "Date") -> bool:
    """Validate date column exists and is datetime."""
    if date_col not in df.columns:
        return False
    return pd.api.types.is_datetime64_any_dtype(df[date_col])


def validate_price_columns(df: pd.DataFrame) -> bool:
    """Validate required price columns exist."""
    required = ["Open", "High", "Low", "Close", "Volume"]
    return all(col in df.columns for col in required)


def validate_no_nulls(df: pd.DataFrame) -> bool:
    """Check for null values."""
    return df.isnull().sum().sum() == 0


def run_validation(df: pd.DataFrame) -> dict:
    """Run all validations and return results."""
    return {
        "date_valid": validate_date_column(df),
        "columns_valid": validate_price_columns(df),
        "no_nulls": validate_no_nulls(df),
    }
