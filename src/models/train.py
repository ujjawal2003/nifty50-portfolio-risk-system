import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib


def get_models() -> dict:
    """Initialize models for comparison."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgboost": XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False),
    }


def train_model(model, X_train, y_train):
    """Train a model."""
    model.fit(X_train, y_train)
    return model


def cross_validate_model(model, X, y, n_splits=5):
    """Perform time series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        scores.append(accuracy_score(y_val, y_pred))
    return scores


def save_model(model, filepath: str):
    """Save trained model to disk."""
    joblib.dump(model, filepath)


def load_model(filepath: str):
    """Load model from disk."""
    return joblib.load(filepath)
