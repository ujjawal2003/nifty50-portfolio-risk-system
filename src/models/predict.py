import pandas as pd
import numpy as np
import joblib


def load_model(filepath: str):
    """Load trained model."""
    return joblib.load(filepath)


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Make predictions using trained model."""
    return model.predict(X)


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Get prediction probabilities."""
    return model.predict_proba(X)


def batch_predict(model, X: pd.DataFrame, batch_size: int = 1000) -> np.ndarray:
    """Make predictions in batches for large datasets."""
    predictions = []
    for i in range(0, len(X), batch_size):
        batch = X.iloc[i:i + batch_size]
        predictions.extend(predict(model, batch))
    return np.array(predictions)
