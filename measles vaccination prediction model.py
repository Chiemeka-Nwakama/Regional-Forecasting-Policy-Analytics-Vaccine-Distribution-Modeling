import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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