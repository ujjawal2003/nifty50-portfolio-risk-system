import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import config as cfg
import components as ui
import data_loader as dl


def render():
    metrics = dl.load_final_metrics()
    cm = dl.load_confusion_matrix()
    deciles = dl.load_deciles()
    calib = dl.load_calibration()
    stability = dl.load_stability()

    ui.section_title(
        "Final Out-of-Sample Evaluation",
        "All numbers below come from the single, locked final test period — never used for "
        "model selection, tuning, or threshold choice.",
    )

    if metrics is not None:
        m = metrics.iloc[0]
        cols = st.columns(4)
        stats = [
            ("ROC-AUC", f"{m['ROC_AUC']:.3f}", "Ranking ability, overall"),
            ("PR-AUC", f"{m['PR_AUC']:.3f}", "Primary selection metric"),
            ("Precision", f"{m['Precision']:.1%}", "Of alerts, % that were real"),
            ("Recall", f"{m['Recall']:.1%}", "Of real events, % caught"),
        ]
        for c, (label, val, sub) in zip(cols, stats):
            with c:
                ui.stat_card(label, val, sub)

        st.write("")
        cols2 = st.columns(4)
        stats2 = [
            ("Accuracy", f"{m['Accuracy']:.1%}", "Not the primary metric"),
            ("Balanced Accuracy", f"{m['Balanced_Accuracy']:.1%}", ""),
            ("F1 Score", f"{m['F1']:.3f}", ""),
            ("Brier Score", f"{m['Brier_Score']:.3f}", "Lower is better-calibrated"),
        ]
        for c, (label, val, sub) in zip(cols2, stats2):
            with c:
                ui.stat_card(label, val, sub)

        ui.info_box(
            f"Test set: <b>{int(m['Test_Observations'])}</b> observations, "
            f"<b>{int(m['Test_Event_Count'])}</b> real drawdown events "
            f"({m['Test_Event_Rate']:.1%} base rate) — a deliberately imbalanced, realistic "
            f"target. This is exactly why PR-AUC, precision and recall matter more here than "
            f"raw accuracy."
        )
    else:
        st.info("`final_test_metrics.csv` not found.")

    st.write("")

    # ---- Confusion matrix + risk deciles side by side ------------------
    c1, c2 = st.columns(2, gap="large")

    with c1:
        ui.section_title("Confusion Matrix", "At the locked alert threshold")
        if cm is not None:
            z = cm.values
            fig = go.Figure(
                data=go.Heatmap(
                    z=z, x=["Predicted: No Event", "Predicted: Event"],
                    y=["Actual: No Event", "Actual: Event"],
                    text=z, texttemplate="%{text}", textfont={"size": 20},
                    colorscale=[[0, "#12172b"], [1, "#4C6FFF"]],
                    showscale=False,
                )
            )
            fig.update_layout(height=320, yaxis_autorange="reversed")
            st.plotly_chart(fig, use_container_width=True, key="perf_confusion_matrix")
        else:
            st.info("Confusion matrix report not found.")

    with c2:
        ui.section_title("Risk Decile Concentration", "Do higher deciles see more real events?")
        if deciles is not None:
            fig = go.Figure(
                go.Bar(
                    x=deciles["Risk_Decile"], y=deciles["Event_Rate"] * 100,
                    marker_color=px.colors.sequential.Blues[2:][: len(deciles)],
                    text=[f"{v:.1%}" for v in deciles["Event_Rate"]],
                    textposition="outside",
                )
            )
            fig.update_layout(
                height=320, xaxis_title="Risk Decile (1 = lowest predicted risk)",
                yaxis_title="Actual Event Rate (%)", xaxis=dict(dtick=1),
            )
            st.plotly_chart(fig, use_container_width=True, key="perf_decile_bar")
        else:
            st.info("Risk decile report not found.")

    st.write("")

    # ---- ROC / PR / Calibration figures --------------------------------
    ui.section_title("Curves", "Pre-generated from the final evaluation notebook")
    t1, t2, t3 = st.columns(3)
    for col, path, caption in [
        (t1, cfg.FIG_ROC, "ROC Curve"),
        (t2, cfg.FIG_PR, "Precision–Recall Curve"),
        (t3, cfg.FIG_CALIBRATION, "Calibration Curve"),
    ]:
        with col:
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)
            else:
                st.info(f"{caption} image not found.")

    if calib is not None:
        with st.expander("📄 View calibration table"):
            st.dataframe(calib, use_container_width=True)

    st.write("")

    # ---- Stability over time --------------------------------------------
    ui.section_title(
        "Chronological Stability",
        "Monthly alert rate & realized event rate across the test period — a strong average "
        "can hide a weak stretch.",
    )
    if stability is not None:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=stability["Month"], y=stability["Alert_Rate"] * 100,
                   name="Alert Rate (%)", marker_color="rgba(76,111,255,0.55)")
        )
        fig.add_trace(
            go.Scatter(x=stability["Month"], y=stability["Event_Rate"] * 100,
                       name="Actual Event Rate (%)", mode="lines+markers",
                       line=dict(color="#e74c3c", width=2))
        )
        fig.update_layout(height=380, yaxis_title="%", xaxis_title="",
                           legend=dict(orientation="h", y=1.1, x=0))
        st.plotly_chart(fig, use_container_width=True, key="perf_stability_chart")
    else:
        st.info("Stability report not found.")

    ui.warn_box(
        "Precision is low in absolute terms (many alerts are false alarms). This is the "
        "expected trade-off of prioritizing recall on a rare, high-impact event — see the "
        "Methodology tab for the reasoning behind the threshold choice."
    )
