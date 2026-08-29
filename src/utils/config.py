import os
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


# Data files
RAW_DATA_FILE = RAW_DATA_DIR / "nifty50_historical_data.csv"
CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "clean_price_data.csv"
FEATURES_FILE = PROCESSED_DATA_DIR / "stock_features.csv"
PORTFOLIO_FILE = PROCESSED_DATA_DIR / "portfolio_dataset.csv"


# Model files
LOGISTIC_MODEL = MODELS_DIR / "logistic_regression.pkl"
RANDOM_FOREST_MODEL = MODELS_DIR / "random_forest.pkl"
XGBOOST_MODEL = MODELS_DIR / "xgboost.pkl"


# Feature columns
FEATURE_COLUMNS = ["Volatility", "Momentum", "RSI", "MACD", "SMA_20", "SMA_50"]
TARGET_COLUMN = "Risk_Label"
