import pandas as pd
import numpy as np
from datetime import datetime


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a number as percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float) -> str:
    """Format a number as Indian Rupees."""
    return f"₹{value:,.2f}"


def get_date_range(df: pd.DataFrame, date_col: str = "Date") -> tuple:
    """Get start and end dates from dataframe."""
    return df[date_col].min(), df[date_col].max()


def log_message(message: str, level: str = "INFO"):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def create_target_label(returns: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Create binary target label based on returns."""
    return (returns > threshold).astype(int)
