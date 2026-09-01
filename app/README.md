# 🛡️ Streamlit App — Quick Start

## Run it

From the **project root** (not inside `app/`):

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## What's inside

| Tab | What it shows |
|---|---|
| 🏠 Overview | Live risk gauge for the most recent date, a live SHAP breakdown of *why*, and full portfolio value/drawdown history |
| 📉 Risk Timeline | Predicted probability vs. real drawdown events across the locked final test period |
| 📈 Model Performance | Final metrics, confusion matrix, risk deciles, ROC/PR/calibration curves, monthly stability |
| 🔎 Explainability | Global SHAP feature importance + the pre-generated SHAP figures from Notebook 10 |
| 🧪 What-If Simulator | Move sliders on key features and watch the real trained model re-predict live, with a live SHAP breakdown |
| 📊 Portfolio | Current sector allocation, per-stock holdings snapshot, and concentration history |
| 📚 Methodology | Target definition, leakage-control principles, and limitations |

## Important notes

- **scikit-learn version is pinned to `1.5.1`** in `requirements.txt`. The saved
  model (`models/09_tuned_selected_model.joblib`) was trained on that version —
  loading it with a much newer scikit-learn (e.g. 1.8.x) raises an unpickling
  error. If you retrain and re-save the model, you can relax this pin.
- The app fixes a data quirk found in `PortfolioWeighted_Volume_Change_1D`,
  which occasionally contains `inf` values on low-volume days. These are
  converted to `NaN` before prediction so the imputer can handle them —
  otherwise the model call raises a `ValueError`.
- This is a **research/monitoring tool**, not a trading system — see the
  Methodology tab for the full list of limitations.

## Tested

The app was smoke-tested headlessly with Streamlit's official `AppTest`
harness across all 7 tabs, all sliders, and the reset button — zero
uncaught exceptions.
