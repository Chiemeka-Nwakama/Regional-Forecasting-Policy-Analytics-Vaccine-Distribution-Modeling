import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
file_path = 'info.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Reshape the data to long format
long_df = pd.melt(df, id_vars=['Country Name', 'Series Name'], var_name='Year', value_name='Value')

# Convert the 'Value' column to numeric, coercing errors to NaN
long_df['Value'] = pd.to_numeric(long_df['Value'], errors='coerce')

long_df['Value'] = long_df.groupby(['Country Name', 'Series Name'])['Value'].transform(lambda x: x.fillna(x.mean()))

# Pivot to wide format with years as rows and predictors as columns
wide_df = long_df.pivot_table(
    index=['Country Name', 'Year'],
    columns='Series Name',
    values='Value'
).reset_index()

# Rename columns for clarity
wide_df.columns.name = None
wide_df.rename(columns={
    'Immunization, measles (% of children ages 12-23 months)': 'Measles_Immunization',
    'Birth rate, crude (per 1,000 people)': 'Birth_Rate',
    'Current health expenditure (% of GDP)': 'Health_Expenditure',
    'Death rate, crude (per 1,000 people)': 'Death_Rate',
    'Mortality rate, under-5 (per 1,000 live births)': 'Under5_Mortality',
    'Number of under-five deaths': 'Under5_Deaths',
    'Out-of-pocket expenditure (% of current health expenditure)': 'OOP_Expenditure',
    'People using at least basic drinking water services (% of population)': 'Basic_Drinking_Water',
    'Population, total': 'Population',
    'GDP growth (annual %)': 'GDP_Growth',
    'Inflation, consumer prices (annual %)': 'Inflation'
}, inplace=True)

# Save the transformed data to a new CSV file (optional)
wide_df.to_csv('transformed_data.csv', index=False)

# Preview the formatted data
print(wide_df)

# Load the transformed data
file_path = 'transformed_data.csv'  # Replace with your transformed data file path
data = pd.read_csv(file_path)

# Drop rows with missing Measles_Immunization or predictors (just to ensure no NaN remains)
data = data.dropna(subset=['Measles_Immunization'])

# Stores per-country forecast values so we can produce the same summary figures
poly_all_country_preds = []

# Loop through each country and perform analysis separately
countries = data['Country Name'].unique()

for country in countries:
    print(f"\nAnalyzing data for {country}")
    
    # Filter data for the current country
    country_data = data[data['Country Name'] == country]
    
    # Select predictors and response variable
    X = country_data.drop(columns=['Country Name', 'Year', 'Measles_Immunization'])
    y = country_data['Measles_Immunization']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- RANDOM FOREST REGRESSOR ---
    rf = RandomForestRegressor(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred_rf = rf.predict(X_test)
    
    # Evaluate Random Forest
    print(f"Random Forest Results for {country}:")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_rf):.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred_rf):.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importance from Random Forest:")
    print(feature_importance)
    
    # --- REGRESSION TREE ---
    dt = DecisionTreeRegressor(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    
    # Predictions
    y_pred_dt = dt.predict(X_test)
    
    # Evaluate Regression Tree
    print(f"\nRegression Tree Results for {country}:")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_dt):.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred_dt):.2f}")
    
    # Visualize the regression tree
    plt.figure(figsize=(20, 10))
    plot_tree(dt, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
    plt.title(f"Regression Tree for {country} - Measles Immunization Prediction")
    plt.show()

    # --- POLYNOMIAL REGRESSION (FORECASTING) ---
    # Fits a polynomial curve to (Year -> Measles_Immunization) for each country,
    # then predicts the next 5 years and produces the same style figures.

    # Ensure Year is numeric and sorted
    country_poly = country_data[['Year', 'Measles_Immunization']].copy()
    country_poly['Year'] = pd.to_numeric(country_poly['Year'], errors='coerce')
    country_poly = country_poly.dropna(subset=['Year', 'Measles_Immunization']).sort_values('Year')

    # Baseline year = last observed year
    baseline_year = int(country_poly['Year'].max())

    # x = years since baseline (x=0 is baseline year)
    x = (country_poly['Year'] - baseline_year).astype(float).values
    y_poly = country_poly['Measles_Immunization'].astype(float).values

    # Polynomial degree (increase carefully to avoid overfitting with limited yearly data)
    poly_degree = 4

    # Fit polynomial coefficients
    coeffs = np.polyfit(x, y_poly, deg=poly_degree)
    poly_fn = np.poly1d(coeffs)

    # Predict next 5 years after baseline
    future_years = np.array([baseline_year + i for i in range(1, 6)])
    future_x = (future_years - baseline_year).astype(float)
    future_pred = poly_fn(future_x)

    # Constrain predictions to realistic bounds
    future_pred = np.clip(future_pred, 0, 99.75)

    # Store predictions for summary plots
    for yr, pred in zip(future_years, future_pred):
        poly_all_country_preds.append({
            'Country': country,
            'Year': int(yr),
            'Predicted_Immunization': float(pred)
        })

    # Per-country forecast plot
    plt.figure(figsize=(10, 5))
    plt.plot(country_poly['Year'].values, y_poly, label='Observed')
    plt.plot(future_years, future_pred, linestyle='--', label='Forecast (Polynomial)')
    plt.title(f"Future Measles Immunization Predictions - {country}")
    plt.xlabel("Year")
    plt.ylabel("Measles Immunization (%)")
    plt.legend()
    plt.show()

# Create the same summary figures across all countries
poly_pred_df = pd.DataFrame(poly_all_country_preds)

if not poly_pred_df.empty:
    # Figure 1: Future predictions for each country
    plt.figure(figsize=(12, 6))
    for c in poly_pred_df['Country'].unique():
        temp = poly_pred_df[poly_pred_df['Country'] == c].sort_values('Year')
        plt.plot(temp['Year'], temp['Predicted_Immunization'], label=c)
    plt.title("Figure 1: Future Measles Immunization Predictions (Polynomial Regression)")
    plt.xlabel("Year")
    plt.ylabel("Predicted Measles Immunization (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Figure_1 future prediction immunzation.png", dpi=300)
    plt.show()

    # Overall trend: average predicted immunization across countries by year
    overall = poly_pred_df.groupby('Year', as_index=False)['Predicted_Immunization'].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(overall['Year'], overall['Predicted_Immunization'])
    plt.title("Overall Trend of Predicted Measles Immunization (Polynomial Regression)")
    plt.xlabel("Year")
    plt.ylabel("Average Predicted Immunization (%)")
    plt.tight_layout()
    plt.savefig("overall trend.png", dpi=300)
    plt.show()
