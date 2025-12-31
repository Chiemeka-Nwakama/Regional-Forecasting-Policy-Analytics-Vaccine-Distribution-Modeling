# Regional Forecasting & Policy Analytics: Vaccine Distribution Modeling

**Authors:**  
Chiemeka Nwakama, Eric Huang, Sam Konstan, Nupur Kumar

---

## Overview

This project analyzes **measles vaccination coverage (MCV1, MCV2)** and **reported measles cases** to understand regional disparities in immunization outcomes and to develop **country-level forecasting models**. Using historical data from global public health sources, the project implements a full analytics pipeline—from data ingestion and cleaning to modeling, evaluation, and forward-looking forecasts—to support **policy planning and resource allocation**.

---

## Key Questions

- How do measles vaccination coverage rates vary across countries and regions?
- Which socioeconomic and health-related factors are most strongly associated with vaccine coverage?
- How well can statistical and machine learning models forecast future vaccination coverage and measles case trends?
- Which countries may require additional intervention based on projected trends?

---

## Data Sources

- **WHO**: Reported measles cases (1974–2023)
- **UNICEF**: MCV1 and MCV2 vaccine coverage (2000–2023)
- **World Bank Open Data**: Demographic, health, and economic indicators (2000–2022)

To ensure consistency, analysis is restricted to overlapping countries and years. Due to data sparsity and heterogeneity at the global level, portions of the analysis focus on specific regions where more reliable comparisons can be made.

---

## Methods

### Data Preparation
- Aligned datasets to a common time window (2000–2023)
- Reformatted wide tables into tidy, long-form structures
- Removed non-informative columns and standardized country identifiers
- Addressed missing data using:
  - Mean imputation
  - Iterative **PCA-based matrix reconstruction** for structured missingness

### Modeling Approaches
- **Log-linear regression** for forecasting reported measles cases
- **Linear regression** for MCV1 and MCV2 vaccine coverage trends
- **Regression trees** and **random forests** for multivariate prediction
- **Polynomial regression** to capture non-linear vaccination trends

### Evaluation
- Models evaluated using:
  - Mean Squared Error (MSE)
  - R-squared
- Predictor importance analyzed to guide feature selection and interpretation

---

## Forecasting

Country-specific models are used to generate **forward-looking projections (2024–2034)** for:
- Reported measles cases
- MCV1 vaccine coverage
- MCV2 vaccine coverage

Forecasts are constrained to realistic bounds (e.g., maximum vaccine coverage capped at 99%) and are produced independently for each country to reflect regional heterogeneity.

---


---

## Key Findings

- Vaccination coverage trends vary substantially across countries, even with similar predictors.
- Child mortality and population-level indicators are among the strongest predictors of vaccine coverage.
- Some countries show increasing projected coverage, while others exhibit stagnation or decline, indicating potential future risk.
- Model performance varies by country, highlighting the limitations of a single global model.

---

## Limitations & Future Work

- Annual data limits the ability to fully capture time-series dependencies.
- Additional predictors (e.g., healthcare access, education, infrastructure) could improve model accuracy.
- Future work could incorporate:
  - Explicit time-series models
  - Deep learning approaches with regularization
  - Subnational data where available

---

## Contributions

| Team Member | Contribution |
|------------|--------------|
| Eric Huang | Modeling, results analysis |
| Sam Konstan | Introduction, discussion, conclusion |
| Nupur Kumar | Data collection and cleaning |
| Chiemeka Nwakama | Modeling, methods, forecasting pipeline |

---

## License

This project is for academic and demonstration purposes. Data sources retain their original licenses.


