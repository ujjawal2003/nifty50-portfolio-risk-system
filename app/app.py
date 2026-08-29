import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import *
from src.utils.helpers import format_percentage, format_currency
from src.features.portfolio_features import calculate_sharpe_ratio, calculate_max_drawdown
from src.models.train import load_model
from components import render_header, render_metric_cards, render_sidebar


def main():
    render_header()
    render_sidebar()

    st.subheader("Portfolio Overview")

    # Load data
    try:
        df = pd.read_csv(CLEAN_DATA_FILE, parse_dates=["Date"])
    except FileNotFoundError:
        st.warning("No data found. Please run the data pipeline first.")
        return

    # Display metrics
    if "Returns" in df.columns:
        returns = df.groupby("Date")["Returns"].mean()
        render_metric_cards(
            total_return=returns.sum(),
            volatility=returns.std() * np.sqrt(252),
            sharpe_ratio=calculate_sharpe_ratio(returns),
            max_drawdown=calculate_max_drawdown(returns),
        )

    # Price chart
    st.subheader("Stock Prices")
    fig = px.line(df, x="Date", y="Close", color="Symbol")
    st.plotly_chart(fig, use_container_width=True)

    # Risk analysis
    st.subheader("Risk Analysis")
    if "Returns" in df.columns:
        fig = px.histogram(df, x="Returns", nbins=50)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
