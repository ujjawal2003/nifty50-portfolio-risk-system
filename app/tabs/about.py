import streamlit as st

import components as ui
import data_loader as dl


def render(bundle):
    ui.section_title("📚 About This System", "")

    st.markdown(
        """
This is a **portfolio risk early-warning system**, not a trading strategy. For each
prediction date, it estimates the probability that the equal-weight NIFTY 50 research
portfolio experiences **at least a 5% drawdown within the next 10 trading days**.
        """
    )

    dist = dl.load_target_distribution()
    if dist is not None:
        ui.section_title("Target Definition", "")
        show = dist.copy()
        show["Positive_Rate_Percent"] = show["Positive_Rate_Percent"].map(lambda v: f"{v:.1f}%")
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption(
            "The 5% / 10-day target was chosen as the primary target — a balance between "
            "event frequency (rare enough to matter, frequent enough to learn from) and "
            "economic significance."
        )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.section_title("🔐 Leakage Control", "")
        st.markdown(
            """
- Features use **only** information available on or before prediction date *t*
- The target describes what happens **after** *t* — never used as a feature
- **Chronological** train / validation / test split — no random shuffling
- The final test period stayed **completely untouched** during model selection,
  hyperparameter tuning, and threshold selection
            """
        )
    with c2:
        ui.section_title("⚙️ Modeling Approach", "")
        if bundle:
            st.markdown(
                f"""
- **Model:** {bundle.get('model_name', 'n/a').replace('_', ' ')}
- **Selection metric:** {bundle.get('selection_metric', 'n/a')}
- **Tuning decision:** `{bundle.get('tuning_decision', 'n/a')}`
- **Reference alert threshold:** {bundle.get('reference_threshold', 0.4):.0%}
- **Random state:** {bundle.get('random_state', 'n/a')}
                """
            )
        st.markdown(
            "Candidates compared: Logistic Regression, Random Forest, HistGradientBoosting. "
            "**PR-AUC** was the primary selection metric because the drawdown event is rare."
        )

    st.write("")
    ui.section_title("⚠️ Limitations", "")
    st.markdown(
        """
| Limitation | Why it matters |
|---|---|
| **Historical dependence** | Learned relationships may not hold in future, unseen market regimes |
| **Class imbalance** | Drawdown events are rare — accuracy alone is a misleading metric here |
| **Probability ≠ certainty** | Outputs are model-estimated risk *scores*, not guaranteed probabilities |
| **No causal inference** | SHAP explains model behaviour, not what actually causes drawdowns |
| **No transaction-cost model** | This is a risk monitor, not a backtested trading strategy |
| **Model risk** | Can fail during unprecedented conditions or structural market breaks |
        """
    )

    ui.warn_box(
        "This tool should <b>not</b> be treated as a standalone buy/sell system. It is intended "
        "as an additional quantitative layer for portfolio risk monitoring, risk budgeting, "
        "and defensive-allocation research."
    )

    with st.expander("🧪 Notebook pipeline (for reference)"):
        st.markdown(
            """
1. Data audit & EDA
2. Price integrity & cleaning
3. Stock-level feature engineering
4. Portfolio construction
5. Risk target creation
6. Portfolio-level feature engineering
7. Model training & evaluation
8. Model comparison & selection
9. Hyperparameter tuning
10. SHAP explainability
11. Final out-of-sample evaluation
            """
        )
