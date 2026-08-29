import pandas as pd
import numpy as np


def normalize_data(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Normalize specified columns using min-max scaling."""
    df_norm = df.copy()
    for col in columns:
        if col in df_norm.columns:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm


def encode_categorical(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """One-hot encode a categorical column."""
    return pd.get_dummies(df, columns=[column], drop_first=True)


def create_sequences(data: np.ndarray, seq_length: int):
    """Create sequences for time series models."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)
