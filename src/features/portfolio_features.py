import pandas as pd
import numpy as np


def calculate_portfolio_returns(weights: np.ndarray, returns: pd.DataFrame) -> pd.Series:
    """Calculate portfolio returns given weights."""
    return (returns * weights).sum(axis=1)


def calculate_portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    """Calculate portfolio volatility."""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """Calculate Sharpe ratio."""
    excess_returns = returns.mean() * 252 - risk_free_rate
    volatility = returns.std() * np.sqrt(252)
    return excess_returns / volatility if volatility != 0 else 0


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_beta(portfolio_returns: pd.Series, market_returns: pd.Series) -> float:
    """Calculate portfolio beta."""
    covariance = np.cov(portfolio_returns, market_returns)[0][1]
    variance = market_returns.var()
    return covariance / variance if variance != 0 else 0
