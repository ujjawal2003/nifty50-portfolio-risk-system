# Final Results — NIFTY 50 Portfolio Risk Prediction System

## 1. Executive Summary

This project develops an AI-based early-warning system for predicting elevated portfolio drawdown risk in a diversified NIFTY 50 portfolio.

The primary prediction target is:

> **Whether the portfolio will experience at least a 5% drawdown within the next 10 trading days.**

The modelling pipeline uses point-in-time portfolio and stock-level features, chronological model development, validation-based model selection, time-aware hyperparameter tuning, SHAP explainability, and a final untouched test-period evaluation.

The system is designed as a **risk-monitoring and early-warning framework**, not as a standalone trading strategy.

---

## 2. Research Question

The central research question is:

> Can historical portfolio-level and cross-sectional market information provide useful early warning of an upcoming material portfolio drawdown?

The project evaluates this question using a supervised binary classification framework.

---

## 3. Portfolio Construction

The portfolio is constructed as a daily-rebalanced equal-weight portfolio using the eligible NIFTY 50 stock universe.

Prediction dates require sufficient stock coverage to avoid relying on an excessively sparse early sample.

The primary eligibility rule established during target construction was:

**Minimum 40 stocks with valid same-day returns.**

---

## 4. Prediction Target

Three candidate future drawdown targets were investigated:

| Target | Definition |
|---|---|
| 3% Drawdown | At least 3% portfolio drawdown within next 10 trading days |
| 5% Drawdown | At least 5% portfolio drawdown within next 10 trading days |
| 10% Drawdown | At least 10% portfolio drawdown within next 10 trading days |

The final project uses:

### Primary Target

**5% portfolio drawdown within the next 10 trading days.**

The 5% target provides a useful balance between event frequency and economic significance.

---

## 5. Feature Engineering

The model uses information available at prediction date `t`.

Feature families include:

### Stock-level features

- 1-day returns
- 5-day returns
- 10-day returns
- 20-day returns
- 60-day returns
- rolling volatility
- moving averages
- price-to-moving-average ratios
- drawdown measures
- volume and liquidity indicators
- intraday range
- ATR
- RSI
- market-relative measures

### Portfolio-level features

- portfolio return
- portfolio volatility
- current drawdown
- rolling returns
- rolling maximum drawdown
- stock concentration
- largest stock weight
- top-5 stock weight
- sector concentration
- sector HHI

### Cross-sectional features

Stock-level signals are aggregated using the portfolio's same-day weights.

This allows the model to capture both portfolio-level risk and the internal state of the portfolio.

---

## 6. Leakage Control

A central design principle is:

> **Features use information available by date `t`; targets describe what happens after date `t`.**

Future target variables are excluded from the feature matrix.

Chronological ordering is preserved throughout model development.

No random train/test shuffling is used.

The final test period is reserved for out-of-sample evaluation.

---

## 7. Model Development

The initial model comparison considered:

- Logistic Regression
- Random Forest
- HistGradientBoosting

PR-AUC was selected as the primary model-selection metric because the positive drawdown event is relatively infrequent.

ROC-AUC, precision, recall, F1 and Brier score were retained as supporting diagnostics.

The development workflow was:

```text
Training data
     ↓
Candidate model comparison
     ↓
Validation model selection
     ↓
Random Forest candidate
     ↓
Time-aware hyperparameter tuning
     ↓
Validation comparison
     ↓
Final selected model

## 8. Hyperparameter Tuning

Random Forest hyperparameters were evaluated using time-aware cross-validation within the training period.

The final test period was not used during hyperparameter selection.

The tuning process evaluated parameters including:

number of estimators
maximum tree depth
minimum samples per leaf
maximum feature selection
class weighting

Tuning was accepted only when it improved the development-period objective.

The final tuning decision was recorded in:

reports/hyperparameter_tuning_summary.csv

and the selected model artifact is:

models/09_tuned_selected_model.joblib

9. Alert Threshold

The reference operational threshold is:

0.40 predicted probability.

The threshold was established using development/validation information.

The final test period was not used to optimize this threshold.

This means that a test observation is classified as an alert when:

Predicted probability >= 0.40

10. Explainability

SHAP analysis was performed to understand which portfolio features contribute to model predictions.

The explainability outputs include:

global feature importance
SHAP beeswarm analysis
local explanation examples

Available figures:

reports/figures/shap_global_feature_importance.png
reports/figures/shap_beeswarm.png
reports/figures/shap_local_explanation_1.png
reports/figures/shap_local_explanation_2.png
reports/figures/shap_local_explanation_3.png

The purpose of SHAP is not to establish causal relationships.

Instead, it explains how the trained model uses available features when generating risk predictions.

11. Final Out-of-Sample Evaluation

Notebook 11 performs the final evaluation on the previously untouched test period.

The final evaluation reports:

ROC-AUC
PR-AUC
Accuracy
Balanced Accuracy
Precision
Recall
F1
Brier Score
Confusion matrix
ROC curve
Precision-Recall curve
Calibration
Monthly stability
Risk-decile concentration

The authoritative numerical results are stored in:

reports/final_test_metrics.csv

The observation-level predictions are stored in:

reports/final_test_predictions.csv

12. Final Test Metrics

The final metrics should be taken directly from:

reports/final_test_metrics.csv
| Metric            |            Final Test Result |
| ----------------- | ---------------------------: |
| ROC-AUC           | See `final_test_metrics.csv` |
| PR-AUC            | See `final_test_metrics.csv` |
| Accuracy          | See `final_test_metrics.csv` |
| Balanced Accuracy | See `final_test_metrics.csv` |
| Precision         | See `final_test_metrics.csv` |
| Recall            | See `final_test_metrics.csv` |
| F1                | See `final_test_metrics.csv` |
| Brier Score       | See `final_test_metrics.csv` |
The final test metrics are intentionally referenced from the generated report rather than manually hard-coded here. This ensures that the documented results remain consistent with the executed final evaluation.

13. Confusion Matrix

The final confusion matrix is stored in:

reports/final_confusion_matrix.csv

It evaluates the model using the locked probability threshold.

The four outcomes are:

True Positive — correctly identified future drawdown event
False Positive — warning issued without the defined drawdown event
True Negative — correctly identified non-event
False Negative — drawdown event not identified by the warning threshold

The confusion matrix provides a practical view of the model's warning behaviour beyond aggregate metrics such as ROC-AUC.

14. Precision-Recall Analysis

The Precision-Recall curve is particularly important for this project because the positive drawdown event is relatively infrequent.

The final curve is available at:

reports/figures/final_test_precision_recall_curve.png

PR-AUC should therefore be interpreted alongside ROC-AUC rather than relying on accuracy alone.

A model can achieve high accuracy simply by predicting the majority non-event class. Precision, recall and PR-AUC provide a more meaningful assessment of its ability to identify actual drawdown events.

15. Calibration

Probability calibration is evaluated using:

reports/figures/final_test_calibration.png

and:

reports/final_calibration_summary.csv

Calibration provides evidence about whether higher predicted probabilities correspond reasonably to higher observed event frequencies.

The model should therefore be interpreted as producing model-estimated risk scores, rather than guaranteed probabilities of future market events.

16. Risk Decile Analysis

Test observations are divided into predicted-risk deciles.

The resulting analysis is stored in:

reports/final_risk_deciles.csv

The purpose is to determine whether observations assigned higher predicted risk contain a greater concentration of actual drawdown events.

A useful risk model should ideally show increasing event rates as predicted risk increases.

This analysis provides an intuitive way to evaluate whether the model meaningfully ranks observations from relatively lower-risk to higher-risk periods.

17. Test-Period Stability

Performance is also examined across chronological test-period blocks.

Results are stored in:

reports/final_test_stability.csv

This analysis is important because an average performance metric can hide periods in which the model performs poorly.

The stability analysis is descriptive and does not modify the final model.

A robust risk model should ideally maintain useful discriminatory behaviour across different market environments rather than relying entirely on a single historical regime.

18. Final System Architecture

Historical NIFTY 50 Data
          │
          ▼
Data Quality & Cleaning
          │
          ▼
Stock-Level Features
          │
          ▼
Portfolio Construction
          │
          ▼
Future Drawdown Target
          │
          ▼
Portfolio-Level Feature Engineering
          │
          ▼
Chronological Train / Validation / Test Split
          │
          ▼
Model Comparison
          │
          ▼
Time-Aware Hyperparameter Tuning
          │
          ▼
Selected Random Forest
          │
          ▼
SHAP Explainability
          │
          ▼
Locked Alert Threshold
          │
          ▼
Final Out-of-Sample Evaluation
          │
          ▼
Portfolio Risk Early Warning


19. Interpretation

The system should be interpreted as an early-warning risk model.

For a given prediction date:

Model Inputs
     ↓
Predicted Risk Probability
     ↓
Risk Threshold
     ↓
Normal / Elevated Risk Alert

A high predicted probability indicates that the model considers the defined 5% drawdown event more likely relative to lower-scored observations.

It does not imply:

certainty of a drawdown
guaranteed trading losses
guaranteed trading profits
causal relationships between individual features and market movements

The intended use is therefore to provide an additional quantitative signal for portfolio risk monitoring.


20. Limitations

Several limitations should be considered.

Historical dependence

The model is trained on historical market behaviour and may perform differently under future market regimes.

Class imbalance

The drawdown event is less frequent than the non-event class, making PR-AUC more informative than accuracy alone.

Probability interpretation

Model probabilities should be treated as risk scores unless calibration is demonstrated sufficiently strongly.

Market regime changes

Relationships between volatility, momentum, concentration and future drawdowns can change over time.

No causal inference

Feature importance and SHAP values describe model behaviour, not economic causality.

No transaction-cost model

This project focuses on portfolio risk prediction rather than execution or trading profitability.

Portfolio assumptions

The portfolio construction methodology uses historical constituent and pricing information and therefore may not perfectly reproduce every real-world implementation detail of an investable NIFTY 50 portfolio.

Model risk

Machine-learning models can fail during unusual market conditions, structural breaks or events that are poorly represented in the training data.


21. Reproducibility

The complete workflow is implemented through sequential notebooks:

01 → Data audit and EDA
02 → Price integrity and cleaning
03 → Stock feature engineering
04 → Portfolio construction
05 → Risk target creation
06 → Portfolio feature engineering
07 → Model training and evaluation
08 → Model comparison and selection
09 → Hyperparameter tuning
10 → SHAP explainability
11 → Final model evaluation

The final selected model is stored as:

models/09_tuned_selected_model.joblib

The final test predictions and evaluation reports are stored under:

reports/

Important generated outputs include:

reports/final_test_metrics.csv
reports/final_test_predictions.csv
reports/final_confusion_matrix.csv
reports/final_calibration_summary.csv
reports/final_risk_deciles.csv
reports/final_test_stability.csv
reports/hyperparameter_search_results.csv
reports/hyperparameter_tuning_summary.csv
reports/tuned_model_threshold_analysis.csv
reports/tuned_model_cv_folds.csv

The project is designed so that the modelling process can be reproduced by executing the notebooks sequentially.


## 22. Conclusion

This project demonstrates an end-to-end machine-learning workflow for portfolio drawdown-risk prediction.

The system combines historical NIFTY 50 data, stock-level signals, portfolio-level risk measures, concentration metrics and market information to estimate the likelihood of a material future portfolio drawdown.

The key methodological principles are:

Point-in-time feature construction
Explicit future drawdown target
Chronological model development
Validation-based model selection
Time-aware hyperparameter tuning
Locked alert threshold
SHAP-based interpretability
Untouched final test evaluation

The final system is best viewed as a portfolio risk early-warning framework that can potentially support portfolio monitoring, risk budgeting and defensive decision-making.

It should not be interpreted as a standalone trading strategy or as a guarantee of future market performance.

The most important outcome of the project is therefore not simply a single model score, but the development of a reproducible, leakage-controlled and interpretable portfolio risk prediction pipeline from raw market data through final out-of-sample evaluation.