import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as cfg
import components as ui
import data_loader as dl

# Friendly labels + slider config for the features exposed to the user.
# (min/max/step are set generously around realistic historical ranges.)
SLIDER_FEATURES = [
    ("Volatility_20D", "Portfolio Volatility (20D)", 0.0, 0.08, 0.001, "{:.1%}"),
    ("Current_Drawdown", "Current Drawdown", -0.60, 0.0, 0.005, "{:.1%}"),
    ("PortfolioWeighted_ATR_14D_Pct", "Avg. True Range (14D, %)", 0.0, 0.08, 0.001, "{:.1%}"),
    ("PortfolioWeighted_Drawdown_20D", "Weighted Drawdown (20D)", -0.5, 0.0, 0.005, "{:.1%}"),
    ("PortfolioWeighted_RSI_14D", "Weighted RSI (14D)", 0.0, 100.0, 1.0, "{:.0f}"),
    ("Top_5_Stock_Weight", "Top-5 Stock Weight", 0.05, 0.5, 0.01, "{:.1%}"),
    ("Largest_Sector_Weight", "Largest Sector Weight", 0.05, 0.6, 0.01, "{:.1%}"),
    ("Return_20D", "Portfolio Return (20D)", -0.3, 0.3, 0.005, "{:.1%}"),
]


def render(bundle, ml_df: pd.DataFrame):
    ui.section_title(
        "🧪 What-If Risk Simulator",
        "Nudge key portfolio conditions and see how the model's risk estimate reacts. "
        "Everything else stays fixed at the most recent snapshot.",
    )

    latest = ml_df.iloc[[-1]].copy()

    if "sim_reset" not in st.session_state:
        st.session_state["sim_reset"] = 0

    reset_col, _ = st.columns([1, 5])
    with reset_col:
        if st.button("↺ Reset to latest snapshot"):
            st.session_state["sim_reset"] += 1

    left, right = st.columns([1, 1.3], gap="large")

    sim_values = {}
    with left:
        for feat, label, lo, hi, step, fmt in SLIDER_FEATURES:
            default = float(latest[feat].iloc[0])
            default = min(max(default, lo), hi)
            key = f"slider_{feat}_{st.session_state['sim_reset']}"
            val = st.slider(label, min_value=float(lo), max_value=float(hi),
                             value=default, step=float(step), key=key,
                             format=None)
            sim_values[feat] = val
            st.caption(f"Latest actual: {fmt.format(default)}")

    sim_row = latest.copy()
    for feat, val in sim_values.items():
        sim_row[feat] = val

    proba = dl.predict_proba(bundle, sim_row)[0]
    baseline_proba = dl.predict_proba(bundle, latest)[0]
    threshold = bundle.get("reference_threshold", cfg.DEFAULT_THRESHOLD)

    with right:
        ui.section_title("Simulated Risk", "")
        st.plotly_chart(ui.risk_gauge(proba, threshold), use_container_width=True, key="sim_gauge")
        st.markdown(ui.risk_pill(proba), unsafe_allow_html=True)

        delta = proba - baseline_proba
        arrow = "🔺" if delta > 0 else ("🔻" if delta < 0 else "➖")
        st.metric(
            "vs. Latest Actual Snapshot",
            f"{proba:.1%}",
            f"{arrow} {delta:+.1%}",
        )

        try:
            feats, sv, xvals = dl.shap_values_for_row(bundle, sim_row)
            contrib = (
                pd.DataFrame({"feature": feats, "shap": sv})
                .assign(abs_shap=lambda d: d["shap"].abs())
                .sort_values("abs_shap", ascending=False)
                .head(8)
                .sort_values("shap")
            )
            fig = go.Figure(
                go.Bar(
                    x=contrib["shap"], y=contrib["feature"], orientation="h",
                    marker_color=["#e74c3c" if v > 0 else "#2ecc71" for v in contrib["shap"]],
                    text=[f"{v:+.3f}" for v in contrib["shap"]], textposition="outside",
                )
            )
            fig.update_layout(height=320, xaxis_title="SHAP impact", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True, key="sim_shap_bar")
        except Exception as e:
            st.caption(f"SHAP breakdown unavailable: {e}")

    ui.info_box(
        "This simulator reuses the real trained pipeline (imputer + Random Forest) — "
        "predictions update live. It's for building intuition about the model's "
        "sensitivities, not a forecasting tool."
    )
