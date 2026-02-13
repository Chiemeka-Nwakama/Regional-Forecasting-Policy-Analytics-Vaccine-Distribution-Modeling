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

### 2) Modeling: Predicting Measles Immunization Coverage (Python)

Target (response variable):
- **Measles_Immunization** = “Immunization, measles (% of children ages 12–23 months)”

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

Models trained **separately per country**:
- **Random Forest Regressor** (performance + feature importance)
- **Regression Tree (DecisionTreeRegressor)** (interpretability; visualized with a shallow max depth)

### 3) Evaluation and Interpretation

For each country/model:
- **Mean Squared Error (MSE)**
- **R-squared (R²)**

Interpretability:
- **Random Forest feature importances** used to identify influential predictors
- Regression tree plots used to explain decision logic (“if-then” splits)

---

## Forecasting & Visualizations

This repository includes:
- Country-level immunization projections based on fitted trend models
- Regression tree diagrams (per country) for interpretability
- A combined regional trend visualization

See the figures at the bottom of this README.

---

## Repository Outputs

- `transformed_data.csv`: tidy country-year dataset produced from the raw indicator table
- Printed tables of model performance (MSE, R²) per country
- Feature importance rankings per country (Random Forest)
- Regression tree plots per country
- Forecast figures for future immunization trends

---

## Notes / Limitations

- **Time dependence:** the ML split uses random train/test sampling; a time-based split (train early years → test later years) would better respect time-series structure.
- **Missingness:** mean imputation is simple; more robust approaches (interpolation, iterative imputation) may improve stability.
- **Country heterogeneity:** results vary by country; a single global model may not generalize well without additional structure.

---

## How to Run (high level)

1. Place raw input files in the project directory (or update paths in scripts).
2. Run the Python data transformation and modeling script to:
   - generate `transformed_data.csv`
   - train Random Forest + Regression Tree per country
   - print evaluation metrics and feature importances
   - render regression tree plots
3. Run the R script to:
   - clean WHO/UNICEF time series
   - fit trend models and generate forward projections

---

## Figures (stored in the repo root)

### Regression trees (per country)
![Cameroon Regression Tree](figures/Cameroon%20Reg%20Tree.png)
![Ghana Regression Tree](figures/Ghana%20Reg%20Tree.png)
![Liberia Regression Tree](figures/Liberia%20Reg%20Tree.png)
![Nigeria Regression Tree](figures/Nigeria%20Reg%20Tree.png)
![Sierra Leone Regression Tree](figures/Sierra%20Leone%20Reg%20Tree.png)

### Forecast plots
![Figure 1: Future Immunization Predictions](figures/Figure__1_future_prediction_immunzation.png)
![Overall Trend](figures/overall%20trend.png)
