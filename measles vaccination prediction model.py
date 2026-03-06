import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load the transformed data
file_path = 'transformed_data.csv'
data = pd.read_csv(file_path)

# Keep rows where the target exists
data = data.dropna(subset=['Measles_Immunization'])

# Ensure Year is numeric
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
data = data.dropna(subset=['Year'])

# Sort for proper time-based feature creation
data = data.sort_values(['Country Name', 'Year']).copy()

# Create lag and rolling features within each country
data['Immunization_Lag1'] = data.groupby('Country Name')['Measles_Immunization'].shift(1)
data['Immunization_Lag2'] = data.groupby('Country Name')['Measles_Immunization'].shift(2)
data['Immunization_Rolling3'] = (
    data.groupby('Country Name')['Measles_Immunization']
    .shift(1)
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)

# Stores per-country forecast values for summary figures
ridge_all_country_preds = []

# Stores results for comparison across countries
all_results = []

# Smaller predictor set for the two-step model
# This is simpler and more stable keeping important variables
predictor_cols = [
     'Birth_Rate',
        'Health_Expenditure',
        'Death_Rate',
        'GDP_Growth',
        'Inflation',
        'OOP_Expenditure',
        'Basic_Drinking_Water',
]

# Ridge values to try
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

# Loop through each country and perform analysis separately
countries = data['Country Name'].unique()

for country in countries:
    print(f"\nAnalyzing data for {country}")

    # Filter data for the current country
    country_data = data[data['Country Name'] == country].copy()
    country_data = country_data.sort_values('Year')

    # Drop early rows that do not yet have lag values
    country_data_model = country_data.dropna(
        subset=['Immunization_Lag1', 'Immunization_Lag2', 'Immunization_Rolling3']
    ).copy()

    unique_years = np.sort(country_data_model['Year'].unique())
    if len(unique_years) < 8:
        print(f"Skipping {country}: not enough yearly data points for a stable time-based split.")
        continue

    cutoff_idx = int(len(unique_years) * 0.8) - 1
    cutoff_year = unique_years[cutoff_idx]

    train_df = country_data_model[country_data_model['Year'] <= cutoff_year].copy()
    test_df = country_data_model[country_data_model['Year'] > cutoff_year].copy()

    if train_df.shape[0] < 5 or test_df.shape[0] < 2:
        print(f"Skipping {country}: train/test too small after time split.")
        continue

    # Center year for more stable regression
    year_base = country_data_model['Year'].min()
    country_data_model['Year_Centered'] = country_data_model['Year'] - year_base
    train_df['Year_Centered'] = train_df['Year'] - year_base
    test_df['Year_Centered'] = test_df['Year'] - year_base

    # Include Year and lagged target features so the model can learn time behavior
    feature_cols = [
        'Year_Centered',
        'Birth_Rate',
        'Health_Expenditure',
        'Death_Rate',
        'GDP_Growth',
        'Inflation',
        'Under5_Mortality',
        'Under5_Deaths',
        'OOP_Expenditure',
        'Basic_Drinking_Water',
        'Population',
        'Immunization_Lag1',
        'Immunization_Lag2',
        'Immunization_Rolling3'
    ]

    # Keep only features that actually exist in the dataframe
    feature_cols = [col for col in feature_cols if col in train_df.columns]

    X_train = train_df[feature_cols]
    y_train = train_df['Measles_Immunization']
    X_test = test_df[feature_cols]
    y_test = test_df['Measles_Immunization']

    # Impute using training-only statistics to avoid leakage
    imputer = SimpleImputer(strategy='mean')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # --- NAIVE BASELINE ---
    # Predict each test year using the previous year's actual immunization value
    y_pred_naive = test_df['Immunization_Lag1'].values

    naive_mse = mean_squared_error(y_test, y_pred_naive)
    naive_r2 = r2_score(y_test, y_pred_naive)

    print(f"Naive Baseline Results for {country}:")
    print(f"Mean Squared Error: {naive_mse:.2f}")
    print(f"R2 Score: {naive_r2:.2f}")

    # --- RANDOM FOREST REGRESSOR ---
    rf = RandomForestRegressor(
        random_state=42,
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=2
    )
    rf.fit(X_train_imp, y_train)

    y_pred_rf = rf.predict(X_test_imp)

    rf_mse = mean_squared_error(y_test, y_pred_rf)
    rf_r2 = r2_score(y_test, y_pred_rf)

    print(f"\nRandom Forest Results for {country}:")
    print(f"Mean Squared Error: {rf_mse:.2f}")
    print(f"R2 Score: {rf_r2:.2f}")

    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importance from Random Forest:")
    print(feature_importance)

    # --- REGRESSION TREE ---
    dt = DecisionTreeRegressor(
        max_depth=3,
        min_samples_leaf=2,
        random_state=42
    )
    dt.fit(X_train_imp, y_train)

    y_pred_dt = dt.predict(X_test_imp)

    dt_mse = mean_squared_error(y_test, y_pred_dt)
    dt_r2 = r2_score(y_test, y_pred_dt)

    print(f"\nRegression Tree Results for {country}:")
    print(f"Mean Squared Error: {dt_mse:.2f}")
    print(f"R2 Score: {dt_r2:.2f}")

    plt.figure(figsize=(20, 10))
    plot_tree(dt, feature_names=feature_cols, filled=True, rounded=True, fontsize=10)
    plt.title(f"Regression Tree for {country} - Measles Immunization Prediction")
    plt.show()

    # TWO-STEP RIDGE: BACKTEST and FORECAST
    ridge_train_mse = np.nan
    ridge_test_mse = np.nan
    ridge_train_rmse = np.nan
    ridge_test_rmse = np.nan
    ridge_train_r2 = np.nan
    ridge_test_r2 = np.nan
    best_alpha = np.nan

    if country_data_model.shape[0] >= 12:
        holdout_k = 5
        train_ridge = country_data_model.iloc[:-holdout_k].copy()
        test_ridge = country_data_model.iloc[-holdout_k:].copy()

        backtest_base = train_ridge['Year'].min()
        train_ridge['Year_Centered'] = train_ridge['Year'] - backtest_base
        test_ridge['Year_Centered'] = test_ridge['Year'] - backtest_base

        y_train_ridge = train_ridge['Measles_Immunization']
        y_test_ridge = test_ridge['Measles_Immunization']

        # Step 1: forecast each predictor for held-out years using training history only
        forecasted_test_predictors = pd.DataFrame({
            'Year': test_ridge['Year'].values,
            'Year_Centered': test_ridge['Year_Centered'].values
        })

        best_predictor_models = {}

        for pred_col in predictor_cols:
            if pred_col not in train_ridge.columns:
                continue

            pred_hist = train_ridge[['Year_Centered', pred_col]].dropna().copy()

            if pred_hist.shape[0] >= 3:
                best_pred_model = None
                best_pred_alpha = None
                best_pred_mse = np.inf

                pred_train_split = pred_hist.iloc[:-1].copy() if pred_hist.shape[0] >= 4 else pred_hist.copy()
                pred_valid_split = pred_hist.iloc[-1:].copy() if pred_hist.shape[0] >= 4 else pred_hist.iloc[-1:].copy()

                for alpha in alphas:
                    pred_model = Pipeline([
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('scaler', StandardScaler()),
                        ('ridge', Ridge(alpha=alpha))
                    ])

                    pred_model.fit(pred_train_split[['Year_Centered']], pred_train_split[pred_col])
                    pred_valid_pred = pred_model.predict(pred_valid_split[['Year_Centered']])
                    pred_valid_mse = mean_squared_error(pred_valid_split[pred_col], pred_valid_pred)

                    if pred_valid_mse < best_pred_mse:
                        best_pred_mse = pred_valid_mse
                        best_pred_model = pred_model
                        best_pred_alpha = alpha

                best_predictor_models[pred_col] = (best_pred_model, best_pred_alpha)
                pred_future = best_pred_model.predict(forecasted_test_predictors[['Year_Centered']])
                forecasted_test_predictors[pred_col] = pred_future
            else:
                fallback_value = train_ridge[pred_col].dropna().iloc[-1] if train_ridge[pred_col].dropna().shape[0] > 0 else np.nan
                forecasted_test_predictors[pred_col] = fallback_value
                best_predictor_models[pred_col] = (None, None)

        # Step 2: fit final immunization model using observed training rows
        ridge_feature_cols = ['Year_Centered'] + [col for col in predictor_cols if col in train_ridge.columns]

        X_train_ridge = train_ridge[ridge_feature_cols]
        X_test_ridge = forecasted_test_predictors[ridge_feature_cols]

        best_ridge_model = None
        best_test_mse = np.inf
        best_train_pred = None
        best_test_pred = None

        for alpha in alphas:
            ridge_model = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=alpha))
            ])

            ridge_model.fit(X_train_ridge, y_train_ridge)

            y_train_pred_ridge = np.clip(ridge_model.predict(X_train_ridge), 0, 99.75)
            y_test_pred_ridge = np.clip(ridge_model.predict(X_test_ridge), 0, 99.75)

            current_test_mse = mean_squared_error(y_test_ridge, y_test_pred_ridge)

            if current_test_mse < best_test_mse:
                best_test_mse = current_test_mse
                best_alpha = alpha
                best_ridge_model = ridge_model
                best_train_pred = y_train_pred_ridge
                best_test_pred = y_test_pred_ridge

        ridge_train_mse = mean_squared_error(y_train_ridge, best_train_pred)
        ridge_test_mse = mean_squared_error(y_test_ridge, best_test_pred)
        ridge_train_rmse = np.sqrt(ridge_train_mse)
        ridge_test_rmse = np.sqrt(ridge_test_mse)
        ridge_train_r2 = r2_score(y_train_ridge, best_train_pred)
        ridge_test_r2 = r2_score(y_test_ridge, best_test_pred)
        ridge_test_residuals = y_test_ridge - best_test_pred

        print(f"\nTwo-Step Ridge Backtest for {country} (hold out last {holdout_k} years):")
        print(f"Best Alpha: {best_alpha}")
        print(f"Train MSE: {ridge_train_mse:.2f}")
        print(f"Test MSE: {ridge_test_mse:.2f}")
        print(f"Train RMSE: {ridge_train_rmse:.2f}")
        print(f"Test RMSE: {ridge_test_rmse:.2f}")
        print(f"Train R2: {ridge_train_r2:.2f}")
        print(f"Test R2: {ridge_test_r2:.2f}")
        print("Test residuals:", ridge_test_residuals.values)

        final_ridge = best_ridge_model.named_steps['ridge']
        ridge_coef_df = pd.DataFrame({
            'Feature': ridge_feature_cols,
            'Coefficient': final_ridge.coef_
        }).sort_values(by='Coefficient', key=lambda s: np.abs(s), ascending=False)

        print("\nTwo-Step Ridge Coefficients:")
        print(ridge_coef_df)
        print(f"\nIntercept: {final_ridge.intercept_:.4f}")

        plt.figure(figsize=(10, 5))
        plt.plot(train_ridge['Year'], y_train_ridge, label='Train Actual')
        plt.plot(test_ridge['Year'], y_test_ridge, 'o-', label='Test Actual')
        plt.plot(test_ridge['Year'], best_test_pred, 's--', label='Test Predicted')
        plt.title(f"Two-Step Ridge Holdout Performance - {country}")
        plt.xlabel("Year")
        plt.ylabel("Measles Immunization (%)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.axhline(y=0, linestyle='--')
        plt.scatter(test_ridge['Year'], ridge_test_residuals)
        plt.title(f"Residual Plot for Two-Step Ridge - {country}")
        plt.xlabel("Year")
        plt.ylabel("Residual (Actual - Predicted)")
        plt.tight_layout()
        plt.show()

        # Refit predictor models on full history and forecast next 5 years
        full_ridge = country_data_model.copy()
        full_base = full_ridge['Year'].min()
        full_ridge['Year_Centered'] = full_ridge['Year'] - full_base

        future_years = np.array([int(full_ridge['Year'].max()) + i for i in range(1, 6)])
        future_year_centered = future_years - full_base

        future_predictor_df = pd.DataFrame({
            'Year': future_years,
            'Year_Centered': future_year_centered
        })

        for pred_col in predictor_cols:
            if pred_col not in full_ridge.columns:
                continue

            pred_hist_full = full_ridge[['Year_Centered', pred_col]].dropna().copy()

            if pred_hist_full.shape[0] >= 3:
                best_pred_model_full = None
                best_pred_mse_full = np.inf

                pred_train_split = pred_hist_full.iloc[:-1].copy() if pred_hist_full.shape[0] >= 4 else pred_hist_full.copy()
                pred_valid_split = pred_hist_full.iloc[-1:].copy() if pred_hist_full.shape[0] >= 4 else pred_hist_full.iloc[-1:].copy()

                for alpha in alphas:
                    pred_model_full = Pipeline([
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('scaler', StandardScaler()),
                        ('ridge', Ridge(alpha=alpha))
                    ])

                    pred_model_full.fit(pred_train_split[['Year_Centered']], pred_train_split[pred_col])
                    pred_valid_pred = pred_model_full.predict(pred_valid_split[['Year_Centered']])
                    pred_valid_mse = mean_squared_error(pred_valid_split[pred_col], pred_valid_pred)

                    if pred_valid_mse < best_pred_mse_full:
                        best_pred_mse_full = pred_valid_mse
                        best_pred_model_full = pred_model_full

                pred_future = best_pred_model_full.predict(future_predictor_df[['Year_Centered']])
                future_predictor_df[pred_col] = pred_future
            else:
                fallback_value = full_ridge[pred_col].dropna().iloc[-1] if full_ridge[pred_col].dropna().shape[0] > 0 else np.nan
                future_predictor_df[pred_col] = fallback_value

        # Final immunization model on full observed history
        X_full_ridge = full_ridge[ridge_feature_cols]
        y_full_ridge = full_ridge['Measles_Immunization']

        best_ridge_model_full = None
        best_full_alpha = None
        best_full_mse = np.inf

        eval_train_split = full_ridge.iloc[:-1].copy() if full_ridge.shape[0] >= 4 else full_ridge.copy()
        eval_valid_split = full_ridge.iloc[-1:].copy() if full_ridge.shape[0] >= 4 else full_ridge.iloc[-1:].copy()

        X_eval_train = eval_train_split[ridge_feature_cols]
        y_eval_train = eval_train_split['Measles_Immunization']
        X_eval_valid = eval_valid_split[ridge_feature_cols]
        y_eval_valid = eval_valid_split['Measles_Immunization']

        for alpha in alphas:
            model_full = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=alpha))
            ])

            model_full.fit(X_eval_train, y_eval_train)
            y_valid_pred = np.clip(model_full.predict(X_eval_valid), 0, 99.75)
            valid_mse = mean_squared_error(y_eval_valid, y_valid_pred)

            if valid_mse < best_full_mse:
                best_full_mse = valid_mse
                best_full_alpha = alpha
                best_ridge_model_full = model_full

        best_ridge_model_full.fit(X_full_ridge, y_full_ridge)

        X_future_ridge = future_predictor_df[ridge_feature_cols]
        future_coverage = np.clip(best_ridge_model_full.predict(X_future_ridge), 0, 99.75)

        print(f"Forecast alpha used for {country}: {best_full_alpha}")

        for yr, pred in zip(future_years, future_coverage):
            ridge_all_country_preds.append({
                'Country': country,
                'Year': int(yr),
                'Predicted_Immunization': float(pred)
            })

        plt.figure(figsize=(10, 5))
        plt.plot(full_ridge['Year'], full_ridge['Measles_Immunization'], label='Observed')
        plt.plot(future_years, future_coverage, linestyle='--', marker='o', label='Forecast (Two-Step Ridge)')
        plt.title(f"Future Measles Immunization Predictions - {country}")
        plt.xlabel("Year")
        plt.ylabel("Measles Immunization (%)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    else:
        print(f"\nSkipping two-step ridge backtest for {country}: not enough data.")

    all_results.append({
        'Country': country,
        'Naive_MSE': naive_mse,
        'Naive_R2': naive_r2,
        'RF_MSE': rf_mse,
        'RF_R2': rf_r2,
        'DT_MSE': dt_mse,
        'DT_R2': dt_r2,
        'TwoStepRidge_Best_Alpha': best_alpha,
        'TwoStepRidge_Train_MSE': ridge_train_mse,
        'TwoStepRidge_Test_MSE': ridge_test_mse,
        'TwoStepRidge_Train_R2': ridge_train_r2,
        'TwoStepRidge_Test_R2': ridge_test_r2
    })

# Comparison table across countries
results_df = pd.DataFrame(all_results)
print("\nModel Comparison Across Countries:")
print(results_df)

results_df.to_csv("model_comparison_results.csv", index=False)

# Create the summary figures across all countries
ridge_pred_df = pd.DataFrame(ridge_all_country_preds)

if not ridge_pred_df.empty:
    plt.figure(figsize=(12, 6))
    for c in ridge_pred_df['Country'].unique():
        temp = ridge_pred_df[ridge_pred_df['Country'] == c].sort_values('Year')
        plt.plot(temp['Year'], temp['Predicted_Immunization'], label=c)
    plt.title("Figure 1: Future Measles Immunization Predictions (Two-Step Ridge)")
    plt.xlabel("Year")
    plt.ylabel("Predicted Measles Immunization (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Figure_1_future_prediction_immunization.png", dpi=300)
    plt.show()

    overall = ridge_pred_df.groupby('Year', as_index=False)['Predicted_Immunization'].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(overall['Year'], overall['Predicted_Immunization'])
    plt.title("Overall Trend of Predicted Measles Immunization (2023-2027) in West African Countries")
    plt.xlabel("Year")
    plt.ylabel("Average Predicted Immunization (%)")
    plt.tight_layout()
    plt.savefig("overall_trend.png", dpi=300)
    plt.show()

# Grouped bar chart for Test MSE
if not results_df.empty:
    countries = results_df['Country']
    x = np.arange(len(countries))
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5 * width, results_df['Naive_MSE'], width, label='Naive')
    plt.bar(x - 0.5 * width, results_df['RF_MSE'], width, label='Random Forest')
    plt.bar(x + 0.5 * width, results_df['DT_MSE'], width, label='Decision Tree')
    plt.bar(x + 1.5 * width, results_df['TwoStepRidge_Test_MSE'], width, label='Two-Step Ridge')

    plt.xticks(x, countries, rotation=45)
    plt.ylabel("Mean Squared Error")
    plt.title("Test MSE Comparison Across Models")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Grouped bar chart for Test R2
if not results_df.empty:
    countries = results_df['Country']
    x = np.arange(len(countries))
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5 * width, results_df['Naive_R2'], width, label='Naive')
    plt.bar(x - 0.5 * width, results_df['RF_R2'], width, label='Random Forest')
    plt.bar(x + 0.5 * width, results_df['DT_R2'], width, label='Decision Tree')
    plt.bar(x + 1.5 * width, results_df['TwoStepRidge_Test_R2'], width, label='Two-Step Ridge')

    plt.xticks(x, countries, rotation=45)
    plt.ylabel("R2 Score")
    plt.title("Test R2 Comparison Across Models")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Actual + predicted graph using values already in your dataframes

plt.figure(figsize=(12.8, 7.2))

# Plot actual historical values from the original data
for country in data['Country Name'].unique():
    actual_df = data[data['Country Name'] == country].sort_values('Year')
    plt.plot(
        actual_df['Year'],
        actual_df['Measles_Immunization'],
        label=f'{country} (Actual)'
    )

# Plot predicted future values from your forecast dataframe
# Use ridge_pred_df if you are running the two-step ridge code
if not ridge_pred_df.empty:
    for country in ridge_pred_df['Country'].unique():
        pred_df = ridge_pred_df[ridge_pred_df['Country'] == country].sort_values('Year')
        plt.plot(
            pred_df['Year'],
            pred_df['Predicted_Immunization'],
            linestyle='--',
            label=f'{country} (Predicted)'
        )

plt.title('Future Measles Immunization Predictions')
plt.xlabel('Year')
plt.ylabel('Measles Immunization (%)')
plt.legend(loc='lower center')
plt.tight_layout()
plt.show()