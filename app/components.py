import streamlit as st
import plotly.graph_objects as go


def render_header():
    """Render app header."""
    st.set_page_config(page_title="Nifty 50 Portfolio Risk System", layout="wide")
    st.title("Nifty 50 Portfolio Risk System")
    st.markdown("---")


def render_sidebar():
    """Render sidebar controls."""
    with st.sidebar:
        st.header("Controls")
        st.selectbox("Select Model", ["XGBoost", "Random Forest", "Logistic Regression"], key="model")
        st.date_input("Start Date", key="start_date")
        st.date_input("End Date", key="end_date")
        st.button("Run Analysis", key="run_analysis")


def render_metric_cards(total_return: float, volatility: float, sharpe_ratio: float, max_drawdown: float):
    """Render metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{total_return:.2%}")
    col2.metric("Volatility", f"{volatility:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    col4.metric("Max Drawdown", f"{max_drawdown:.2%}")


def render_stock_selector(symbols: list):
    """Render stock selection dropdown."""
    return st.multiselect("Select Stocks", symbols, default=symbols[:5])


def render_loading_spinner(message: str = "Loading..."):
    """Render loading spinner."""
    return st.spinner(message)
