import plotly.graph_objects as go
import streamlit as st

import config as cfg
import components as ui
import data_loader as dl


def render():
    ui.section_title(
        "🔎 Global Explainability (SHAP)",
        "What drives the model's predictions across the whole dataset, on average.",
    )

    shap_df = dl.load_shap_importance()
    if shap_df is not None:
        top = shap_df.sort_values("Mean_Absolute_SHAP", ascending=False).head(15).iloc[::-1]
        fig = go.Figure(
            go.Bar(
                x=top["Mean_Absolute_SHAP"], y=top["Feature"], orientation="h",
                marker_color="#4C6FFF",
                text=[f"{v:.3f}" for v in top["Mean_Absolute_SHAP"]],
                textposition="outside",
            )
        )
        fig.update_layout(
            height=520, xaxis_title="Mean |SHAP value| (average impact on risk score)",
            yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True, key="explain_global_bar")
        st.caption(
            "Ranked by average absolute impact on the predicted probability across all "
            "test observations. Volatility and drawdown-related features dominate — "
            "consistent with financial intuition."
        )
    else:
        st.info("`shap_feature_importance.csv` not found.")

    st.write("")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.section_title("Global Feature Importance", "")
        if cfg.FIG_SHAP_GLOBAL.exists():
            st.image(str(cfg.FIG_SHAP_GLOBAL), use_container_width=True)
        else:
            st.info("Image not found.")
    with c2:
        ui.section_title("SHAP Beeswarm", "Direction & magnitude across observations")
        if cfg.FIG_SHAP_BEESWARM.exists():
            st.image(str(cfg.FIG_SHAP_BEESWARM), use_container_width=True)
        else:
            st.info("Image not found.")

    st.write("")
    ui.section_title("Local Explanations", "Individual example predictions, from the notebook")
    cols = st.columns(3)
    for col, path in zip(cols, cfg.FIG_SHAP_LOCAL):
        with col:
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.info("Image not found.")

    ui.info_box(
        "SHAP explains <b>model behaviour</b> — which features the model leaned on for a "
        "given prediction — not <b>causation</b>. A feature can be genuinely predictive "
        "without directly causing future drawdowns."
    )
