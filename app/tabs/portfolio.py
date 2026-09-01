import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as cfg
import components as ui
import data_loader as dl


def render():
    ui.section_title(
        "📊 Portfolio Composition",
        "Equal-weight NIFTY 50 research portfolio — latest constituent snapshot.",
    )

    stock_w = dl.load_stock_weights()
    sector_w = dl.load_sector_weights()
    stock_feats = dl.load_stock_features()

    if stock_w is None or sector_w is None:
        st.info("Weight files not found.")
        return

    latest_date = stock_w["Date"].max()
    latest_w = stock_w[stock_w["Date"] == latest_date].copy()
    latest_sector = sector_w[sector_w["Date"] == latest_date].copy().sort_values(
        "Sector_Weight", ascending=False
    )

    st.caption(f"Snapshot date: {latest_date:%d %b %Y} · {len(latest_w)} eligible stocks")

    c1, c2 = st.columns([1, 1.2], gap="large")

    with c1:
        ui.section_title("Sector Allocation", "")
        colors = [cfg.SECTOR_COLORS.get(s, "#4C6FFF") for s in latest_sector["Sector"]]
        fig = go.Figure(
            go.Pie(
                labels=latest_sector["Sector"], values=latest_sector["Sector_Weight"],
                hole=0.55, marker=dict(colors=colors),
                textinfo="label+percent", textfont=dict(size=11),
            )
        )
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key="portfolio_sector_pie")

    with c2:
        ui.section_title("Holdings Snapshot", "Latest price, momentum & risk signals per stock")
        if stock_feats is not None:
            latest_sf = stock_feats[stock_feats["Date"] == stock_feats["Date"].max()]
            merged = latest_w[["Ticker", "Sector", "Weight"]].merge(
                latest_sf[["Ticker", "Close", "Return_1D", "RSI_14D", "Volatility_20D"]],
                on="Ticker", how="left",
            )
            merged = merged.sort_values("Sector")
            show = merged[["Ticker", "Sector", "Close", "Return_1D", "RSI_14D", "Volatility_20D"]].copy()
            show.columns = ["Ticker", "Sector", "Close (₹)", "1D Return", "RSI (14D)", "Volatility (20D)"]
            show["1D Return"] = show["1D Return"].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "—")
            show["Volatility (20D)"] = show["Volatility (20D)"].map(lambda v: f"{v:.2%}" if pd.notna(v) else "—")
            show["RSI (14D)"] = show["RSI (14D)"].map(lambda v: f"{v:.0f}" if pd.notna(v) else "—")
            show["Close (₹)"] = show["Close (₹)"].map(lambda v: f"{v:,.1f}" if pd.notna(v) else "—")
            st.dataframe(show, use_container_width=True, height=420, hide_index=True)
        else:
            st.dataframe(latest_w[["Ticker", "Sector", "Weight"]], use_container_width=True, height=420)

    st.write("")
    ui.section_title("Concentration Over Time", "Top-5 stock weight & largest sector weight, historical")
    ml_df = dl.load_ml_features()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ml_df["Date"], y=ml_df["Top_5_Stock_Weight"] * 100,
                               name="Top-5 Stock Weight (%)", line=dict(color="#4C6FFF")))
    fig2.add_trace(go.Scatter(x=ml_df["Date"], y=ml_df["Largest_Sector_Weight"] * 100,
                               name="Largest Sector Weight (%)", line=dict(color="#00C2A8")))
    fig2.update_layout(height=340, yaxis_title="%", xaxis_title="",
                        legend=dict(orientation="h", y=1.1, x=0))
    st.plotly_chart(fig2, use_container_width=True, key="portfolio_concentration_line")
