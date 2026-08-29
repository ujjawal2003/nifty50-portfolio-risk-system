import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_classifier(y_true, y_pred) -> dict:
    """Evaluate classification model performance."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Get confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true, y_pred) -> str:
    """Get detailed classification report."""
    return classification_report(y_true, y_pred)


def compare_models(results: dict) -> pd.DataFrame:
    """Compare results from multiple models."""
    return pd.DataFrame(results).T
