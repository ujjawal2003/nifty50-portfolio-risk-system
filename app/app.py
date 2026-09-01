"""
NIFTY 50 Portfolio Risk Prediction System — Streamlit App
Run from the project root with:  streamlit run app/app.py
"""
import sys
from pathlib import Path

# Make sibling modules (config, data_loader, components, styling, tabs/*)
# importable regardless of the working directory streamlit is launched from.
APP_DIR = Path(__file__).resolve().parent
for p in (str(APP_DIR), str(APP_DIR / "tabs")):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

st.set_page_config(
    page_title="NIFTY 50 Portfolio Risk System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import config as cfg
import data_loader as dl
import styling
import components as ui

import overview
import timeline
import performance
import explainability
import simulator
import portfolio
import about


def sidebar(bundle, ml_df):
    with st.sidebar:
        st.markdown("### 🛡️ Risk System")
        st.caption("NIFTY 50 Portfolio Early-Warning")
        st.divider()

        if bundle is not None and ml_df is not None:
            latest = ml_df.iloc[[-1]]
            proba = dl.predict_proba(bundle, latest)[0]
            label, color = dl.risk_band(proba)
            st.markdown("**Live Risk Status**")
            st.markdown(ui.risk_pill(proba), unsafe_allow_html=True)
            st.caption(f"As of {ml_df['Date'].iloc[-1]:%d %b %Y}")
        st.divider()

        st.markdown("**Files detected**")
        checks = [
            ("Model artifact", cfg.MODEL_FILE.exists()),
            ("Feature dataset", cfg.ML_FEATURES_FILE.exists()),
            ("Final test predictions", cfg.FINAL_PREDICTIONS_FILE.exists()),
            ("SHAP importance report", cfg.SHAP_IMPORTANCE_FILE.exists()),
        ]
        for label, ok in checks:
            st.markdown(f"{'✅' if ok else '❌'} {label}")

        st.divider()
        st.caption(
            "⚠️ Research & monitoring tool only — not a trading signal, "
            "not financial advice."
        )


def main():
    styling.inject_css()
    styling.apply_plotly_theme()

    bundle = dl.load_model()
    ml_df = dl.load_ml_features() if cfg.ML_FEATURES_FILE.exists() else None

    ui.hero_header()
    sidebar(bundle, ml_df)

    if bundle is None:
        st.error(
            f"Model file not found at `{cfg.MODEL_FILE}`. Make sure the notebooks have "
            "been run and the model artifact exists before launching the app."
        )
        return
    if ml_df is None:
        st.error(
            f"Feature dataset not found at `{cfg.ML_FEATURES_FILE}`. Run the pipeline "
            "notebooks first (01 through 06 at minimum)."
        )
        return

    tab_labels = [
        "🏠 Overview",
        "📉 Risk Timeline",
        "📈 Model Performance",
        "🔎 Explainability",
        "🧪 What-If Simulator",
        "📊 Portfolio",
        "📚 Methodology",
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        overview.render(bundle, ml_df)
    with tabs[1]:
        timeline.render(bundle)
    with tabs[2]:
        performance.render()
    with tabs[3]:
        explainability.render()
    with tabs[4]:
        simulator.render(bundle, ml_df)
    with tabs[5]:
        portfolio.render()
    with tabs[6]:
        about.render(bundle)


if __name__ == "__main__":
    main()
