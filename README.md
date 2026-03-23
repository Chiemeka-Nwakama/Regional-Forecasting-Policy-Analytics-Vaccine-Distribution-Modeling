# Regional Forecasting & Policy Analytics: Vaccine Distribution Modeling

**Authors:** Chiemeka Nwakama, Eric Huang, Sam Konstan, Nupur Kumar

---

## Overview

This project analyzes **measles vaccination coverage (MCV1, MCV2)** and **reported measles cases** to understand disparities in immunization outcomes and to build **country-level models** that support public health planning. The repository includes an end-to-end workflow: **data ingestion → cleaning/reshaping → modeling → evaluation → forecasting/visualization**, with a focused case study in **West Africa** where cross-country comparisons are more meaningful and data coverage is more consistent.

---

## Key Questions

- How do measles vaccination coverage rates vary across countries and over time?
- Which socioeconomic and health indicators are most associated with measles immunization coverage?
- How well do tree-based ML models predict measles immunization coverage at the country level?
- How much does prior immunization history improve prediction when lagged and rolling time features are included?
- Which countries may require additional intervention based on projected trends?

---

## Data Sources

- **WHO**: Reported measles cases (1974–2023)
- **UNICEF**: MCV1 and MCV2 coverage (2000–2023)
- **World Bank Open Data**: Demographic, health, and economic indicators (2000–2022)

To ensure comparability, analysis is restricted to **overlapping years/countries** where applicable. Because global comparisons can be confounded by large structural differences and missingness, the multivariate modeling portion of this project focuses on **West Africa** (Cameroon, Ghana, Liberia, Nigeria, Sierra Leone), selected based on data availability.

---

## Methods

### 1) Data Preparation (R + Python)

**R (WHO/UNICEF time-aligned trends)**
- Trimmed measles case data to align with vaccine coverage years (2000–2023)
- Cleaned missing/blank values and standardized year column names
- Generated country-level forecasts using:
  - **Log-linear regression** for measles cases (fit in log-space, then back-transformed)
  - **Linear regression** for MCV1 and MCV2 coverage trends (with coverage capped at 99%)

**Python (World Bank predictors for coverage modeling)**
- Reshaped raw World Bank-style tables from wide → long (`melt`) and long → wide (`pivot_table`)
- Converted values to numeric and addressed missingness using **group-wise mean imputation**
  - (filled missing values per **Country × Indicator** using that group’s mean)
- Standardized feature names for readable modeling columns
- Sorted records by **Country Name** and **Year** so time-based features could be created correctly within each country
- Performed Augmented Dickey-Fuller (ADF) tests on the target and numeric predictor variables within each country to check whether series were stationary before modeling and forecasting
- Engineered temporal immunization history features to capture persistence in vaccine coverage:
  - **Immunization_Lag1** = previous year’s measles immunization value
  - **Immunization_Lag2** = immunization value from two years earlier
  - **Immunization_Rolling3** = rolling 3-year mean of prior immunization levels, shifted so only past information is used
- Centered the year variable within each country (**Year_Centered**) to improve numerical stability for regression-based forecasting models

### 2) Modeling: Predicting Measles Immunization Coverage (Python)

Target (response variable):
- **Measles_Immunization** = “Immunization, measles coverage %

Predictors (features):
- Birth_Rate
- Health_Expenditure
- Death_Rate
- Under5_Mortality
- Under5_Deaths
- OOP_Expenditure
- Basic_Drinking_Water
- Population
- GDP_Growth
- Inflation
- Year_Centered
- Immunization_Lag1
- Immunization_Lag2
- Immunization_Rolling3

Models trained **separately per country**:
- **Naive baseline**
  - predicts each test year using the previous year’s observed immunization value
  - provides a simple benchmark for whether more complex models outperform persistence alone
- **Random Forest Regressor** (performance + feature importance)
- **Regression Tree (DecisionTreeRegressor)** (interpretability; visualized with a shallow max depth)
- **Two-Step Ridge Forecast Models** for forward prediction:
  - A first set of **Ridge regression models** is fit to forecast future values of key country-level predictors such as birth rate, health expenditure, GDP growth, inflation, out-of-pocket expenditure, and access to basic drinking water.
  - A second **Ridge regression model** then uses those forecasted predictors, along with centered year, to estimate future measles immunization coverage.
  - This two-model setup separates **predictor forecasting** from **coverage prediction**, helping stabilize forecasts under multicollinearity and small-sample country-level settings.
  - The ridge framework also helps regularize coefficients when socioeconomic variables are correlated and country sample sizes are limited.

Modeling setup:
- Data are modeled **country by country** rather than pooling all countries into one global model
- Early rows without sufficient lag history are excluded from lag-based modeling
- A **time-based split** is used for the main predictive modeling, with earlier years used for training and later years used for testing
- Missing feature values are imputed using **training-only means** to avoid data leakage

### 3) Evaluation and Interpretation

For each country/model:
- **Mean Squared Error (MSE)**
- **R-squared (R²)**

Additional evaluation steps:
- The **Naive baseline** is used to check whether machine learning models outperform a simple carry-forward assumption
- The **Two-Step Ridge model** is backtested by holding out the final 5 years, forecasting predictors for those years using only earlier history, and then comparing predicted immunization values with the observed holdout period
- Ridge training and test performance are summarized with:
  - **Train/Test MSE**
  - **Train/Test RMSE**
  - **Train/Test R²**
  - **Residual plots** for holdout years

Interpretability:
- **Random Forest feature importances** used to identify influential predictors
- Regression tree plots used to explain decision logic (“if-then” splits)
- Ridge coefficient tables used to show which forecast inputs have the strongest linear influence on predicted immunization coverage
- Lagged immunization variables help capture momentum and persistence in country vaccine trajectories, which is especially useful when recent coverage levels influence near-term future coverage

---

## Forecasting & Visualizations

This repository includes:
- Country-level immunization projections based on fitted trend models
- Regression tree diagrams (per country) for interpretability
- A combined regional trend visualization
- Forecasted immunization curves generated from the **two-step Ridge framework**, where future feature values are estimated first and then passed into a Ridge coverage model for final prediction
- Holdout performance plots comparing **actual vs predicted** immunization values for the Ridge backtest
- Residual plots for the two-step Ridge backtest
- Regional summary figures showing:
  - predicted future immunization by country
  - average predicted immunization trend across West African countries
  - grouped bar charts comparing **test MSE** and **test R²** across the Naive baseline, Random Forest, Decision Tree, and Two-Step Ridge models
- A combined chart overlaying **historical actual immunization values** with **future predicted values** for each country

See the figures at the bottom of this README.

---

## Repository Outputs

- `transformed_data.csv`: tidy country-year dataset produced from the raw indicator table
- Printed tables of model performance (MSE, R²) per country
- Feature importance rankings per country (Random Forest)
- Regression tree plots per country
- Forecast figures for future immunization trends
- `model_comparison_results.csv`: summary comparison table across countries for Naive, Random Forest, Decision Tree, and Two-Step Ridge models
- Ridge-based future forecasts generated from the two-model prediction pipeline

---

## Notes / Limitations

- **Time dependence:** while the main country models now use a time-based train/test split, forecasting immunization remains challenging because country-level vaccine coverage is serially dependent and can shift abruptly due to policy or outbreak responses.
- **Lag dependence:** lagged immunization variables improve prediction by capturing recent trajectory, but they also mean the model is partly relying on persistence in past coverage levels rather than only external predictors.
- **Missingness:** mean imputation is simple; more robust approaches (interpolation, iterative imputation) may improve stability.
- **Country heterogeneity:** results vary by country; a single global model may not generalize well without additional structure.
- **Forecast uncertainty:** the two-step Ridge approach depends on the quality of projected future predictors, so forecasting error can compound across both stages.


---

## How to Run

1. Place raw input `info.csv` in the project directory (or update paths in scripts).
2. Run the Python data transformation and modeling script to:
   - generate `transformed_data.csv`
   - create lagged immunization and rolling-history features
   - train Naive baseline, Random Forest, and Regression Tree models per country
   - print evaluation metrics and feature importances
   - render regression tree plots
   - fit the two Ridge forecast models for future predictor and immunization prediction
   - backtest the Ridge model on held-out years
   - save `model_comparison_results.csv`
   - output graphs of predicted future immunization trends for the specified forecast horizon

---

## Figures

### Regression trees (per country)
![Cameroon Regression Tree](figures/Cameroon%20Reg%20Tree.png)
![Ghana Regression Tree](figures/Ghana%20Reg%20Tree.png)
![Liberia Regression Tree](figures/Liberia%20Reg%20Tree.png)
![Nigeria Regression Tree](figures/Nigeria%20Reg%20Tree.png)
![Sierra Leone Regression Tree](figures/Sierra%20Leone%20Reg%20Tree.png)

### Forecast plots
![Figure 1: Future Immunization Predictions](figures/Figure_1%20future%20prediction%20immunzation.png)
![Overall Trend](figures/overall%20trend.png)

### Model Performance Comparision (MSE)
![Test MSE Across Models](figures/Test%20MSE%20Across%20Models.png)
