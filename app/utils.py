import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV file."""
    return pd.read_csv(filepath, parse_dates=["Date"])


def filter_by_date(df: pd.DataFrame, start_date, end_date, date_col: str = "Date") -> pd.DataFrame:
    """Filter dataframe by date range."""
    mask = (df[date_col] >= pd.Timestamp(start_date)) & (df[date_col] <= pd.Timestamp(end_date))
    return df.loc[mask]


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Calculate cumulative returns."""
    return (1 + returns).cumprod() - 1


def format_large_number(value: float) -> str:
    """Format large numbers with abbreviations."""
    if abs(value) >= 1e9:
        return f"{value / 1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"{value / 1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"{value / 1e3:.2f}K"
    return f"{value:.2f}"
