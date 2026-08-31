# NIFTY 50 Portfolio Risk Prediction System

An end-to-end machine learning system for predicting the probability of a **5% portfolio drawdown within the next 10 trading days** using historical NIFTY 50 market data, portfolio-level risk features, cross-sectional stock signals, and interpretable machine learning.

The project is designed as a **portfolio risk early-warning system**, not as a standalone trading strategy.

---

## 📌 Project Overview

Financial markets can experience sudden periods of elevated volatility and portfolio drawdowns.

The objective of this project is to build a quantitative early-warning system capable of identifying periods in which the portfolio has an elevated probability of experiencing a material drawdown.

The system follows a complete machine-learning workflow:

```text
Raw NIFTY 50 Data
        ↓
Data Quality Audit
        ↓
Price Cleaning & Integrity Checks
        ↓
Stock-Level Feature Engineering
        ↓
Portfolio Construction
        ↓
Future Drawdown Target Creation
        ↓
Portfolio-Level Feature Engineering
        ↓
Chronological Train / Validation / Test Split
        ↓
Model Comparison
        ↓
Hyperparameter Tuning
        ↓
SHAP Explainability
        ↓
Final Out-of-Sample Evaluation
        ↓
Portfolio Risk Early Warning

🎯 Problem Statement

The project answers the following question:

Can historical portfolio and market information provide useful early warning of an upcoming material portfolio drawdown?

The primary prediction task is binary classification.

For each prediction date t, the model estimates whether the portfolio will experience:

At least a 5% drawdown within the following 10 trading days.

🏆 Primary Target

Three candidate targets were initially investigated:

Target	Horizon
3% drawdown	Next 10 trading days
5% drawdown	Next 10 trading days
10% drawdown	Next 10 trading days

The final target was selected as:

5% Portfolio Drawdown within 10 Trading Days

The target provides a balance between:

event frequency
economic significance
usefulness as a portfolio risk warning


📊 Portfolio Construction

The system constructs a daily portfolio from the eligible NIFTY 50 stock universe.

Prediction dates must contain sufficient valid stock observations.

The final methodology uses:

Minimum 40 eligible stocks

This removes the very sparse early portion of the dataset and improves the reliability of portfolio-level calculations.

The portfolio is constructed using an equal-weight framework.

🧠 Feature Engineering

The project contains multiple feature families.

Stock-Level Features

Examples include:

Daily returns
5-day returns
10-day returns
20-day returns
60-day returns
Rolling volatility
Moving averages
Price-to-moving-average ratios
Drawdown measures
Volume/liquidity indicators
Intraday range
ATR
RSI
Market-relative features
Portfolio-Level Features

The portfolio feature layer includes:

Portfolio return
Portfolio value
Current drawdown
Rolling returns
Rolling volatility
Portfolio concentration
Largest stock weight
Top-5 stock weight
Sector concentration
Sector HHI

Examples:

PortfolioWeighted_Return_1D
PortfolioWeighted_Return_5D
Top_5_Stock_Weight
Largest_Sector_Weight
Sector_HHI
Current_Drawdown
Volatility_5D
Volatility_20D
Volatility_60D


🔐 Leakage Prevention

A major design principle of this project is:

Features use only information available on or before prediction date t.

The target describes what happens after prediction date t.

Therefore:

Information available at t
          ↓
       Features
          ↓
       Model
          ↓
Future period t+1 ... t+10
          ↓
      Drawdown Target

Future target information is not used as a model feature.

The final test period is kept isolated until final evaluation.

⏳ Time-Series Validation

Random train/test splitting is inappropriate for this problem because it can introduce look-ahead bias.

Instead, the project uses chronological splitting.

Historical Data
──────────────────────────────────────────────→ Time

|-------------|-------------|------------------|
     Train       Validation         Test
      70%            15%             15%

The final test period remains untouched during:

model selection
threshold selection
hyperparameter tuning

This provides a more realistic estimate of out-of-sample performance.


🤖 Models

The modelling workflow evaluates multiple classification approaches.

The primary candidates include:

Logistic Regression
Random Forest
HistGradientBoosting

The primary model-selection metric is:

PR-AUC

PR-AUC is emphasized because the positive drawdown event is relatively infrequent.

Supporting metrics include:

ROC-AUC
Accuracy
Balanced Accuracy
Precision
Recall
F1
Brier Score


🌲 Final Model

The model comparison and tuning workflow selected a:

Random Forest classifier

The final model artifact is stored at:

models/09_tuned_selected_model.joblib

The project also maintains the model-selection and tuning outputs under:

reports/


⚙️ Hyperparameter Tuning

Random Forest hyperparameters were evaluated using time-aware cross-validation.

Parameters considered include:

n_estimators
max_depth
min_samples_leaf
max_features
class_weight

The tuning process does not use the final test period for model optimization.

The final tuning decision is documented in:

reports/hyperparameter_tuning_summary.csv

Additional tuning outputs include:

reports/hyperparameter_search_results.csv
reports/tuned_model_cv_folds.csv
reports/tuned_model_threshold_analysis.csv


🚨 Risk Alert Threshold

The operational reference threshold is:

0.40

Therefore:

Predicted probability >= 0.40
                ↓
        Elevated Risk Alert

The threshold was determined using development/validation information rather than optimizing against the final test period.


🔎 Explainability with SHAP

The project uses SHAP to investigate how the final model arrives at its predictions.

The explainability analysis includes:

Global Feature Importance

Identifies the features that have the greatest influence on model predictions.

SHAP Beeswarm

Shows the direction and magnitude of feature contributions across observations.

Local Explanations

Individual observations are analyzed to understand why the model assigned a particular risk score.

Generated figures include:

reports/figures/shap_global_feature_importance.png
reports/figures/shap_beeswarm.png
reports/figures/shap_local_explanation_1.png
reports/figures/shap_local_explanation_2.png
reports/figures/shap_local_explanation_3.png

SHAP results explain model behaviour and should not be interpreted as causal relationships.


📈 Final Evaluation

The final evaluation is performed on the previously untouched test period.

The evaluation includes:

ROC-AUC
PR-AUC
Accuracy
Balanced Accuracy
Precision
Recall
F1
Brier Score
Confusion Matrix
ROC Curve
Precision-Recall Curve
Calibration
Risk Decile Analysis
Chronological Stability Analysis


📊 Final Results

The authoritative final numerical results are generated by Notebook 11.

Key output:

reports/final_test_metrics.csv

Additional final outputs include:

reports/final_test_predictions.csv
reports/final_confusion_matrix.csv
reports/final_calibration_summary.csv
reports/final_risk_deciles.csv
reports/final_test_stability.csv

The project intentionally keeps these values in generated result files rather than manually duplicating them throughout the repository.

This reduces the possibility of documentation and experiment results becoming inconsistent.


📉 Precision-Recall Analysis

Because the drawdown event is relatively uncommon, PR-AUC is particularly important.

The final Precision-Recall curve is available at:

reports/figures/final_test_precision_recall_curve.png

Accuracy alone is not considered sufficient evidence of model usefulness.

A model could obtain high accuracy by predominantly predicting the majority non-event class.

Therefore, the project emphasizes:

PR-AUC
Precision
Recall
F1
Risk Decile Concentration


📐 Calibration

The model's probability outputs are also evaluated for calibration.

Output:

reports/figures/final_test_calibration.png

Summary:

reports/final_calibration_summary.csv

Calibration analysis helps determine whether observations receiving higher predicted probabilities actually experience the target event at higher frequencies.


📊 Risk Decile Analysis

The test observations are ranked according to predicted drawdown probability and divided into risk groups.

Output:

reports/final_risk_deciles.csv

The objective is to determine whether actual drawdown events become increasingly concentrated in higher predicted-risk groups.

Conceptually:

Lowest Risk
     ↓
     ↓
     ↓
Higher Risk
     ↓
     ↓
Highest Risk

A useful risk-ranking model should ideally show increasing event concentration as predicted risk increases.


📅 Stability Analysis

Model performance is also evaluated across chronological portions of the test period.

Output:

reports/final_test_stability.csv

This helps identify whether model behaviour remains reasonably stable across different market conditions.

A strong historical average alone is not sufficient if performance is concentrated entirely within one market regime.


📁 Project Structure
nifty50-portfolio-risk-system/
│
├── app/
│
├── data/
│   ├── external/
│   ├── processed/
│   └── raw/
│
├── models/
│   └── 09_tuned_selected_model.joblib
│
├── notebooks/
│   ├── 01_data_audit_and_eda.ipynb
│   ├── 02_price_integrity_and_cleaning.ipynb
│   ├── 03_stock_feature_engineering.ipynb
│   ├── 04_portfolio_construction.ipynb
│   ├── 05_risk_target_creation.ipynb
│   ├── 06_portfolio_feature_engineering.ipynb
│   ├── 07_model_training_and_evaluation.ipynb
│   ├── 07_time_series_split_and_validation.ipynb
│   ├── 08_model_comparison_and_selection.ipynb
│   ├── 09_hyperparameter_tuning.ipynb
│   ├── 10_shap_explainability.ipynb
│   └── 11_final_model_evaluation.ipynb
│
├── reports/
│   ├── figures/
│   ├── data_quality_report.md
│   ├── final_results.md
│   ├── final_test_metrics.csv
│   ├── final_test_predictions.csv
│   ├── final_confusion_matrix.csv
│   ├── final_calibration_summary.csv
│   ├── final_risk_deciles.csv
│   ├── final_test_stability.csv
│   ├── hyperparameter_search_results.csv
│   ├── hyperparameter_tuning_summary.csv
│   ├── tuned_model_cv_folds.csv
│   └── tuned_model_threshold_analysis.csv
│
├── src/
│   ├── data/
│   │   ├── cleaning.py
│   │   ├── preprocessing.py
│   │   └── validation.py
│   │
│   ├── features/
│   │   ├── portfolio_features.py
│   │   ├── risk_features.py
│   │   └── stock_features.py
│   │
│   ├── models/
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── train.py
│   │
│   └── utils/
│       ├── config.py
│       └── helpers.py
│
├── README.md
└── requirements.txt


🧪 Notebook Workflow

The notebooks form a sequential research pipeline.

Notebook	Purpose
01	Data audit and exploratory analysis
02	Price integrity and cleaning
03	Stock-level feature engineering
04	Portfolio construction
05	Risk target creation
06	Portfolio-level feature engineering
07	Model training and evaluation
07_time_series_split	Time-series validation
08	Model comparison and selection
09	Hyperparameter tuning
10	SHAP explainability
11	Final out-of-sample evaluation


🛠️ Technologies Used

The project is implemented primarily in Python.

Core technologies include:

Python
Pandas
NumPy
Scikit-learn
Matplotlib
SHAP
Jupyter Notebook
Joblib
Git / GitHub


💻 Reproducibility

To reproduce the project:

1. Clone the repository
git clone https://github.com/ujjawal2003/nifty50-portfolio-risk-system.git
2. Enter the project directory
cd nifty50-portfolio-risk-system
3. Install dependencies
pip install -r requirements.txt
4. Run notebooks sequentially

Start from:

01_data_audit_and_eda.ipynb

and continue through:

11_final_model_evaluation.ipynb

The notebooks generate intermediate datasets, model artifacts and reports required by later stages.


🔬 Research Methodology

The project follows the following research discipline:

No random shuffling

Time ordering is preserved.

No test-set optimization

The final test period is used only for final evaluation.

Validation-driven decisions

Model and threshold decisions are made using development data.

Leakage auditing

Future information is explicitly excluded from the feature matrix.

Reproducible outputs

Intermediate and final reports are saved as CSV/Markdown artifacts.

Explainability

SHAP is used to inspect model behaviour.


⚠️ Limitations

This system has several limitations.

Historical Dependence

The model learns relationships from historical data. Future market behaviour may differ.

Regime Changes

Relationships between volatility, momentum, concentration and future drawdowns can change across market regimes.

Class Imbalance

The target event is less frequent than the non-event class.

Probability Interpretation

Model probabilities should be treated as model-generated risk estimates rather than guaranteed probabilities.

No Causal Inference

Feature importance does not imply that a feature causes future drawdowns.

No Transaction-Cost Model

The project focuses on risk prediction rather than trading execution or profitability.

Portfolio Assumptions

The portfolio construction methodology represents a systematic research portfolio and may differ from every real-world NIFTY 50 implementation.

Model Risk

Machine-learning models can fail during unprecedented market conditions or structural breaks.


🚀 Future Improvements

Potential extensions include:

Walk-forward retraining
Probability calibration
Gradient boosting models
XGBoost / LightGBM comparison
Neural-network models
Regime-aware modelling
More sophisticated sector-level features
Macro-economic variables
Volatility-index features
Market breadth indicators
Transaction-cost modelling
Portfolio optimization based on predicted risk
Live monitoring dashboard
Automated daily risk alerts
Model drift monitoring
Production API deployment


💡 Intended Use

The system can potentially be used as an additional quantitative layer for:

Portfolio risk monitoring
Drawdown early warning
Risk budgeting
Defensive allocation decisions
Portfolio stress monitoring
Research into machine-learning-based risk management

It should not be treated as a standalone buy/sell system.


🏁 Conclusion

This project demonstrates a complete end-to-end machine-learning workflow for portfolio drawdown-risk prediction.

The system combines:

Market Data
    +
Stock-Level Signals
    +
Portfolio-Level Risk Features
    +
Concentration Measures
    +
Time-Aware Validation
    +
Machine Learning
    +
SHAP Explainability
    +
Out-of-Sample Evaluation

The final objective is to estimate whether the portfolio is entering a period of elevated drawdown risk.

The most important contribution of the project is not simply a single model metric.

It is the construction of a:

Reproducible, leakage-controlled, interpretable and out-of-sample-tested portfolio risk prediction pipeline.

The resulting framework provides a foundation for further research into portfolio risk forecasting, regime detection and systematic risk management.


📌 Final Project Status

Project: NIFTY 50 Portfolio Risk Prediction System

Primary Target: 5% portfolio drawdown within the next 10 trading days

Portfolio Rule: Minimum 40 eligible stocks

Primary Model: Random Forest

Model Selection Metric: PR-AUC

Reference Risk Threshold: 0.40

Validation: Chronological / time-aware

Explainability: SHAP

Final Evaluation: Untouched chronological test period

Status: ✅ Complete
```
