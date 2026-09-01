import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as cfg
import data_loader as dl
import components as ui


def render(bundle, ml_df: pd.DataFrame):
    latest = ml_df.iloc[[-1]]
    latest_date = latest["Date"].iloc[0]

    proba = dl.predict_proba(bundle, latest)[0]
    threshold = bundle.get("reference_threshold", cfg.DEFAULT_THRESHOLD)
    label, color = dl.risk_band(proba)
    is_alert = proba >= threshold

    # ---- Top row: gauge + headline stats -----------------------------
    col_gauge, col_stats = st.columns([1, 1.6], gap="large")

    with col_gauge:
        ui.section_title("Current Risk Signal", f"As of {latest_date:%d %b %Y}")
        st.plotly_chart(ui.risk_gauge(proba, threshold), use_container_width=True, key="overview_gauge")
        st.markdown(ui.risk_pill(proba), unsafe_allow_html=True)
        if is_alert:
            ui.warn_box(
                f"Predicted probability ({proba:.1%}) is **at or above** the locked "
                f"alert threshold ({threshold:.0%}). The model flags this as an "
                f"**elevated-risk** period for a 5% portfolio drawdown within 10 trading days."
            )
        else:
            ui.info_box(
                f"Predicted probability ({proba:.1%}) is **below** the locked alert "
                f"threshold ({threshold:.0%}). The model currently sees **no elevated** "
                f"drawdown risk signal."
            )

    with col_stats:
        ui.section_title("Portfolio Snapshot", "Equal-weight NIFTY 50 research portfolio")
        row = latest.iloc[0]
        c1, c2 = st.columns(2)
        with c1:
            ui.stat_card("Portfolio Value (indexed)", f"{row['Portfolio_Value']:.2f}", "Base = 1.00")
            ui.stat_card("Current Drawdown", f"{row['Current_Drawdown']:.2%}", "From running peak")
        with c2:
            ui.stat_card("20-Day Volatility", f"{row['Volatility_20D']:.2%}", "Daily, rolling")
            ui.stat_card("Top-5 Stock Weight", f"{row['Top_5_Stock_Weight']:.1%}", "Concentration")

        st.write("")
        c3, c4 = st.columns(2)
        with c3:
            ui.stat_card("Largest Sector Weight", f"{row['Largest_Sector_Weight']:.1%}", "Sector concentration")
        with c4:
            ui.stat_card("Eligible Stocks Today", f"{int(row['Number_of_Stocks'])}", "Min. required: 40")

    st.write("")

    # ---- Why this score: live SHAP for the latest observation --------
    ui.section_title(
        "🔎 Why the model sees it this way",
        "Live SHAP breakdown for today's prediction — top drivers pushing risk up or down.",
    )
    try:
        feats, sv, xvals = dl.shap_values_for_row(bundle, latest)
        contrib = (
            pd.DataFrame({"feature": feats, "shap": sv, "value": xvals})
            .assign(abs_shap=lambda d: d["shap"].abs())
            .sort_values("abs_shap", ascending=False)
            .head(10)
            .sort_values("shap")
        )
        colors = [ui.risk_band(1.0)[1] if v > 0 else "#2ecc71" for v in contrib["shap"]]
        fig = go.Figure(
            go.Bar(
                x=contrib["shap"],
                y=contrib["feature"],
                orientation="h",
                marker_color=["#e74c3c" if v > 0 else "#2ecc71" for v in contrib["shap"]],
                text=[f"{v:+.3f}" for v in contrib["shap"]],
                textposition="outside",
            )
        )
        fig.update_layout(
            height=380,
            xaxis_title="Impact on predicted risk (SHAP value)",
            yaxis_title="",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key="overview_shap_bar")
        st.caption(
            "🔴 Red bars push the risk estimate **up** · 🟢 Green bars pull it **down**. "
            "Explains model behaviour for this prediction only — not a causal claim."
        )
    except Exception as e:
        st.info(f"Live SHAP explanation unavailable in this environment ({e}).")

    st.write("")

    # ---- Portfolio value history --------------------------------------
    ui.section_title("Portfolio Value & Drawdown History", "Full research-portfolio history since inception")
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=ml_df["Date"], y=ml_df["Portfolio_Value"], name="Portfolio Value",
            line=dict(color="#4C6FFF", width=1.6), fill="tozeroy",
            fillcolor="rgba(76,111,255,0.08)",
        )
    )
    fig2.update_layout(height=340, yaxis_title="Indexed Value (Base = 1.0)", xaxis_title="")
    st.plotly_chart(fig2, use_container_width=True, key="overview_portfolio_value")

    fig3 = go.Figure()
    fig3.add_trace(
        go.Scatter(
            x=ml_df["Date"], y=ml_df["Current_Drawdown"] * 100, name="Drawdown",
            line=dict(color="#e74c3c", width=1.2), fill="tozeroy",
            fillcolor="rgba(231,76,60,0.12)",
        )
    )
    fig3.update_layout(height=260, yaxis_title="Drawdown from Peak (%)", xaxis_title="")
    st.plotly_chart(fig3, use_container_width=True, key="overview_drawdown")
