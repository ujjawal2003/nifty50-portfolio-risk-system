"""
All file I/O, caching, and model-inference logic lives here so the
tab modules stay purely about layout/UI.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

import config as cfg

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    """Load the tuned Random Forest model bundle (dict with pipeline + metadata)."""
    if not cfg.MODEL_FILE.exists():
        return None
    bundle = joblib.load(cfg.MODEL_FILE)
    return bundle


def get_feature_columns(bundle):
    return bundle["feature_columns"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading portfolio feature history...")
def load_ml_features():
    df = pd.read_csv(cfg.ML_FEATURES_FILE, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_stock_features():
    df = pd.read_csv(cfg.STOCK_FEATURES_FILE, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    return df


@st.cache_data(show_spinner=False)
def load_stock_weights():
    df = pd.read_csv(cfg.STOCK_WEIGHTS_FILE, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    return df


@st.cache_data(show_spinner=False)
def load_sector_weights():
    df = pd.read_csv(cfg.SECTOR_WEIGHTS_FILE, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    return df


@st.cache_data(show_spinner=False)
def load_csv(path):
    """Generic small-report loader; returns None if the file is missing."""
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_final_metrics():
    return load_csv(cfg.FINAL_METRICS_FILE)


@st.cache_data(show_spinner=False)
def load_final_predictions():
    df = load_csv(cfg.FINAL_PREDICTIONS_FILE)
    if df is not None:
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    return df


@st.cache_data(show_spinner=False)
def load_confusion_matrix():
    df = load_csv(cfg.FINAL_CONFUSION_FILE)
    if df is not None:
        df = df.set_index(df.columns[0])
    return df


@st.cache_data(show_spinner=False)
def load_calibration():
    return load_csv(cfg.FINAL_CALIBRATION_FILE)


@st.cache_data(show_spinner=False)
def load_deciles():
    return load_csv(cfg.FINAL_DECILES_FILE)


@st.cache_data(show_spinner=False)
def load_stability():
    return load_csv(cfg.FINAL_STABILITY_FILE)


@st.cache_data(show_spinner=False)
def load_shap_importance():
    return load_csv(cfg.SHAP_IMPORTANCE_FILE)


@st.cache_data(show_spinner=False)
def load_model_comparison():
    return load_csv(cfg.MODEL_COMPARISON_FILE)


@st.cache_data(show_spinner=False)
def load_target_distribution():
    return load_csv(cfg.RISK_TARGET_DIST_FILE)


@st.cache_data(show_spinner=False)
def load_portfolio_summary():
    return load_csv(cfg.PORTFOLIO_SUMMARY_FILE)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def clean_feature_row(row_df: pd.DataFrame) -> pd.DataFrame:
    """Replace +/-inf with NaN so the imputer can handle it (a known quirk
    in PortfolioWeighted_Volume_Change_1D on low-volume days)."""
    return row_df.replace([np.inf, -np.inf], np.nan)


def predict_proba(bundle, feature_row: pd.DataFrame) -> float:
    """Run the full pipeline (imputer + model) on a single-row (or multi-row)
    feature dataframe and return probability of the positive (drawdown) class."""
    feats = bundle["feature_columns"]
    X = clean_feature_row(feature_row[feats])
    proba = bundle["model"].predict_proba(X)[:, 1]
    return proba


@st.cache_resource(show_spinner=False)
def get_shap_explainer(_bundle):
    """Cached TreeExplainer built directly on the RandomForest step of the
    pipeline (bypasses the imputer, which shap.TreeExplainer doesn't need)."""
    import shap
    rf = _bundle["model"].named_steps["model"]
    return shap.TreeExplainer(rf)


def shap_values_for_row(bundle, feature_row: pd.DataFrame):
    """Return (feature_names, shap_values_for_positive_class, imputed_values)."""
    feats = bundle["feature_columns"]
    X = clean_feature_row(feature_row[feats])
    imputer = bundle["model"].named_steps["imputer"]
    X_imputed = imputer.transform(X)

    explainer = get_shap_explainer(bundle)
    sv = explainer.shap_values(X_imputed)
    sv = np.array(sv)
    if sv.ndim == 3:
        # shape (n_rows, n_features, n_classes) -> take positive class
        sv = sv[:, :, 1]
    return feats, sv[0], X_imputed[0]


def risk_band(prob: float):
    for lo, hi, label, color in cfg.RISK_BANDS:
        if lo <= prob < hi:
            return label, color
    return "High", "#e74c3c"
