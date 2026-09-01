import plotly.graph_objects as go
import streamlit as st

import components as ui
import data_loader as dl


def render(bundle):
    preds = dl.load_final_predictions()
    if preds is None:
        st.info("`final_test_predictions.csv` not found — timeline unavailable.")
        return

    threshold = bundle.get("reference_threshold", 0.40) if bundle else 0.40

    ui.section_title(
        "Predicted Risk Over the Final Test Period",
        "Locked, untouched out-of-sample period — never seen during model selection or tuning.",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ui.stat_card("Test Observations", f"{len(preds):,}")
    with c2:
        ui.stat_card("Real Drawdown Events", f"{int(preds['Actual_Target'].sum())}")
    with c3:
        alerts = int((preds["Predicted_Probability"] >= threshold).sum())
        ui.stat_card("Alerts Raised", f"{alerts}", f"at ≥{threshold:.0%} threshold")
    with c4:
        rate = preds["Actual_Target"].mean()
        ui.stat_card("Base Event Rate", f"{rate:.1%}", "How rare the event is")

    st.write("")

    events = preds[preds["Actual_Target"] == 1]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=preds["Date"], y=preds["Predicted_Probability"],
            name="Predicted Probability", mode="lines",
            line=dict(color="#4C6FFF", width=1.4),
        )
    )
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#f5f7ff",
        annotation_text=f"Alert threshold ({threshold:.0%})",
        annotation_position="top left",
    )
    fig.add_trace(
        go.Scatter(
            x=events["Date"], y=events["Predicted_Probability"],
            name="Actual Drawdown Event", mode="markers",
            marker=dict(color="#e74c3c", size=9, symbol="diamond",
                        line=dict(width=1, color="#fff")),
        )
    )
    fig.update_layout(
        height=440, yaxis_title="Predicted Probability", yaxis_tickformat=".0%",
        xaxis_title="", legend=dict(orientation="h", y=1.08, x=0),
    )
    st.plotly_chart(fig, use_container_width=True, key="timeline_prob_chart")
    st.caption(
        "🔴 Diamonds mark days where a real 5% drawdown actually followed within 10 trading "
        "days. A good early-warning signal should show the line running high just before "
        "those markers."
    )

    with st.expander("📄 View raw prediction log"):
        show = preds.copy()
        show["Predicted_Probability"] = show["Predicted_Probability"].map(lambda v: f"{v:.1%}")
        show["Alert"] = (preds["Predicted_Probability"] >= threshold).map({True: "🚨 Yes", False: "—"})
        st.dataframe(
            show.rename(columns={
                "Actual_Target": "Real Event?",
                "Predicted_Class": "Predicted Class",
            }),
            use_container_width=True, height=360,
        )
