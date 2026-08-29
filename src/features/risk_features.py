import pandas as pd
import numpy as np
from scipy import stats


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Value at Risk using historical method."""
    return np.percentile(returns, (1 - confidence) * 100)


def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Conditional Value at Risk."""
    var = calculate_var(returns, confidence)
    return returns[returns <= var].mean()


def calculate_tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate tracking error."""
    active_returns = portfolio_returns - benchmark_returns
    return active_returns.std() * np.sqrt(252)


def calculate_information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate information ratio."""
    active_returns = portfolio_returns - benchmark_returns
    tracking_error = active_returns.std() * np.sqrt(252)
    excess_return = active_returns.mean() * 252
    return excess_return / tracking_error if tracking_error != 0 else 0


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """Calculate Sortino ratio."""
    excess_returns = returns.mean() * 252 - risk_free_rate
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    return excess_returns / downside_std if downside_std != 0 else 0
