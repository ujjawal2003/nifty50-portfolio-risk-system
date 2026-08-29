import pandas as pd
import numpy as np


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily returns."""
    return prices.pct_change()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns."""
    return np.log(prices / prices.shift(1))


def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Calculate rolling volatility."""
    return returns.rolling(window=window).std() * np.sqrt(252)


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return prices.rolling(window=window).mean()
