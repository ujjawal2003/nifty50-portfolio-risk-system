"""Reusable small UI building blocks used across tabs."""
import streamlit as st
import plotly.graph_objects as go

from data_loader import risk_band


def hero_header():
    st.markdown(
        """
        <div class="hero">
            <h1>🛡️ NIFTY 50 Portfolio Risk Early-Warning System</h1>
            <p>Machine-learning monitor for elevated portfolio drawdown risk — a research
            &amp; monitoring tool, not a trading signal.</p>
            <div class="badge-row">
                <span class="pill">🌲 Random Forest</span>
                <span class="pill">🎯 5% drawdown / 10 trading days</span>
                <span class="pill">⏳ Time-aware validation</span>
                <span class="pill">🔎 SHAP explainable</span>
                <span class="pill">🔒 Locked out-of-sample test</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, sub: str = ""):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_pill(prob: float) -> str:
    label, color = risk_band(prob)
    css_class = {
        "Low": "pill-safe",
        "Moderate": "pill-caution",
        "Elevated": "pill-warning",
        "High": "pill-danger",
    }.get(label, "pill")
    return f'<span class="pill {css_class}">{label} risk · {prob:.1%}</span>'


def section_title(title: str, sub: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="section-sub">{sub}</div>', unsafe_allow_html=True)


def info_box(text: str):
    st.markdown(f'<div class="info-box">ℹ️ {text}</div>', unsafe_allow_html=True)


def warn_box(text: str):
    st.markdown(f'<div class="warn-box">⚠️ {text}</div>', unsafe_allow_html=True)


def risk_gauge(prob: float, threshold: float = 0.40, height: int = 260) -> go.Figure:
    label, color = risk_band(prob)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 40, "color": "#f5f7ff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9aa4b2", "tickfont": {"size": 10}},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "rgba(46, 204, 113, 0.18)"},
                    {"range": [25, 40], "color": "rgba(241, 196, 15, 0.18)"},
                    {"range": [40, 60], "color": "rgba(230, 126, 34, 0.18)"},
                    {"range": [60, 100], "color": "rgba(231, 76, 60, 0.18)"},
                ],
                "threshold": {
                    "line": {"color": "#f5f7ff", "width": 3},
                    "thickness": 0.85,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=10, b=10))
    return fig
