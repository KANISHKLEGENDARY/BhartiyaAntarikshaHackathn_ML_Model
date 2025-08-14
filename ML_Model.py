#Improved PM2.5 Prediction Model with Enhanced Features and Preprocessing
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
import os
import xarray as xr
import warnings
warnings.filterwarnings('ignore')

# ---------- Step 1: Load INSAT AOD from CSV ----------
aod_df = pd.read_csv("aod_data.csv")

# Parse with mixed formats; keep rows that fail as NaT so you can inspect
aod_df['Date'] = pd.to_datetime(
    aod_df['Date'], 
    format='mixed',     # pandas >= 2.0
    dayfirst=True, 
    errors='coerce'
)

bad_aod = aod_df[aod_df['Date'].isna()]
if not bad_aod.empty:
    print("Unparsable AOD dates (fix in CSV):")
    print(bad_aod.head(10))

# ---------- Step 2: Load CPCB PM2.5 ----------
pm_df = pd.read_csv("combined_pm2.5_para.csv")

pm_df['Timestamp'] = pd.to_datetime(
    pm_df['Timestamp'], 
    format='mixed', 
    dayfirst=True, 
    errors='coerce'
)

bad_pm = pm_df[pm_df['Timestamp'].isna()]
if not bad_pm.empty:
    print("Unparsable CPCB timestamps (fix in CSV):")
    print(bad_pm[['Timestamp']].head(10))

pm_df['Date'] = pm_df['Timestamp'].dt.normalize()

# ---------- Step 3: Enhanced MERRA feature extraction ----------
def extract_merra_features_enhanced(nc_folder):
    records = []
    for file in os.listdir(nc_folder):
        if file.endswith(".nc"):
            ds = Dataset(os.path.join(nc_folder, file), 'r')

            time_var = ds.variables['time']
            times = num2date(time_var[:], units=time_var.units, only_use_cftime_datetimes=False)

            # Basic meteorological variables
            ps = ds.variables['PS'][:, 0, 0]
            qv2m = ds.variables['QV2M'][:, 0, 0]
            t2m = ds.variables['T2M'][:, 0, 0]
            ts = ds.variables['TS'][:, 0, 0]
            u10m = ds.variables['U10M'][:, 0, 0]
            qv10m = ds.variables['QV10M'][:, 0, 0]
            slp = ds.variables['SLP'][:, 0, 0]
            t10m = ds.variables['T10M'][:, 0, 0]
            t2mdew = ds.variables['T2MDEW'][:, 0, 0]
            tqi = ds.variables['TQI'][:, 0, 0]
            tql = ds.variables['TQL'][:, 0, 0]

            for i in range(len(times)):
                date_val = times[i]
                if hasattr(date_val, 'year'):
                    date_val = datetime(date_val.year, date_val.month, date_val.day)

                records.append({
                    "Date": date_val,
                    "PS": ps[i],
                    "QV2M": qv2m[i],
                    "T2M": t2m[i],
                    "TS": ts[i],
                    "U10M": u10m[i],
                    "QV10M": qv10m[i],
                    "SLP": slp[i],
                    "T10M": t10m[i],
                    "T2MDEW": t2mdew[i],
                    "TQI": tqi[i],
                    "TQL": tql[i],
                })

    df = pd.DataFrame(records)
    return df.groupby("Date").agg({
        'PS': ['mean', 'std'],
        'QV2M': ['mean', 'std'],
        'T2M': ['mean', 'std'],
        'TS': ['mean', 'std'],
        'U10M': ['mean', 'std'],
        'QV10M': ['mean', 'std'],
        'SLP': ['mean', 'std'],
        'T10M': ['mean', 'std'],
        'T2MDEW': ['mean', 'std'],
        'TQI': ['mean', 'std'],
        'TQL': ['mean', 'std']
    }).reset_index()

merra_df = extract_merra_features_enhanced("merra_downloads")
# Flatten column names
merra_df.columns = ['Date'] + [f"{col[0]}_{col[1]}" for col in merra_df.columns[1:]]

# Convert dates to datetime
aod_df['Date'] = pd.to_datetime(aod_df['Date'])
pm_df['Date'] = pd.to_datetime(pm_df['Date'])
merra_df['Date'] = pd.to_datetime(merra_df['Date'])

# ---------- Step 4: Enhanced Data Preprocessing ----------
# First merge
combined_df = pd.merge(aod_df, pm_df, on="Date", how='inner')
combined_df = pd.merge(combined_df, merra_df, on="Date", how='inner')

# Remove obvious outliers (PM2.5 > 500 or < 0)
combined_df = combined_df[(combined_df['PM2.5 (µg/m³)'] >= 0) & 
                         (combined_df['PM2.5 (µg/m³)'] <= 500)]

print(f"Combined dataset size: {len(combined_df)} records")

# ---------- Step 5: Advanced Feature Engineering ----------
def create_advanced_features(df):
    df = df.copy()
    
    # Time-based features
    df['Month'] = df['Date'].dt.month
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Season'] = df['Month'].map({12: 0, 1: 0, 2: 0,  # Winter
                                   3: 1, 4: 1, 5: 1,   # Spring
                                   6: 2, 7: 2, 8: 2,   # Summer
                                   9: 3, 10: 3, 11: 3}) # Autumn
    
    # Meteorological derived features
    df['Temp_Range'] = df['T2M_mean'] - df['T10M_mean']
    df['Pressure_Gradient'] = df['PS_mean'] - df['SLP_mean']
    df['Humidity_Index'] = df['QV2M_mean'] / (df['T2M_mean'] - df['T2MDEW_mean'] + 1e-3)
    df['Stability_Index'] = df['T2M_mean'] - df['TS_mean']
    df['Moisture_Content'] = df['TQI_mean'] + df['TQL_mean']
    df['Wind_Magnitude'] = np.sqrt(df['U10M_mean']**2)
    
    # Interaction features with AOD
    df['AOD_Temp'] = df['Mean_AOD'] * df['T2M_mean']
    df['AOD_Humidity'] = df['Mean_AOD'] * df['QV2M_mean']
    df['AOD_Pressure'] = df['Mean_AOD'] * df['PS_mean']
    df['AOD_Wind'] = df['Mean_AOD'] * df['Wind_Magnitude']
    
    # Atmospheric stability indicators
    df['Atmospheric_Stability'] = (df['PS_mean'] * df['QV2M_mean']) / (df['T2M_mean'] + 273.15)
    df['Mixing_Height_Proxy'] = df['T2M_mean'] / df['PS_mean'] * 1000
    
    # Variability features (using std columns)
    df['Temp_Variability'] = df['T2M_std']
    df['Pressure_Variability'] = df['PS_std']
    df['Humidity_Variability'] = df['QV2M_std']
    
    # Polynomial features for key variables
    df['AOD_squared'] = df['Mean_AOD'] ** 2
    df['Temp_squared'] = df['T2M_mean'] ** 2
    df['Humidity_squared'] = df['QV2M_mean'] ** 2
    
    return df

# Apply feature engineering
feature_df = create_advanced_features(combined_df)

# ---------- Step 6: Feature Selection and Scaling ----------
# Define base features and engineered features
base_features = [col for col in feature_df.columns if col.endswith('_mean') or col == 'Mean_AOD']
engineered_features = ['Month', 'DayOfYear', 'Season', 'Temp_Range', 'Pressure_Gradient',
                      'Humidity_Index', 'Stability_Index', 'Moisture_Content', 'Wind_Magnitude',
                      'AOD_Temp', 'AOD_Humidity', 'AOD_Pressure', 'AOD_Wind',
                      'Atmospheric_Stability', 'Mixing_Height_Proxy', 'Temp_Variability',
                      'Pressure_Variability', 'Humidity_Variability', 'AOD_squared',
                      'Temp_squared', 'Humidity_squared']

all_features = base_features + engineered_features

# Clean dataset
clean_df = feature_df.dropna(subset=all_features + ['PM2.5 (µg/m³)'])
print(f"Clean dataset size: {len(clean_df)} records")

# Prepare features and target
X = clean_df[all_features].copy()
y = clean_df['PM2.5 (µg/m³)']

# Handle infinite values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Feature selection using statistical tests
selector = SelectKBest(score_func=f_regression, k=min(25, len(all_features)))
X_selected = selector.fit_transform(X, y)
selected_features = [all_features[i] for i in selector.get_support(indices=True)]
print(f"Selected {len(selected_features)} best features")

# Scale features
scaler = RobustScaler()  # More robust to outliers than StandardScaler
X_scaled = scaler.fit_transform(X_selected)

# ---------- Step 7: Enhanced Model Training with Hyperparameter Tuning ----------
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5)
)

# Try multiple models with hyperparameter tuning
models = {
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [200, 300, 400],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [200, 300, 400],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10]
        }
    }
}

best_model = None
best_score = -np.inf
best_name = ""

for name, config in models.items():
    print(f"\nTuning {name}...")
    
    # Reduced parameter grid for faster execution
    if name == 'RandomForest':
        param_grid = {
            'n_estimators': [300],
            'max_depth': [15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', None]
        }
    else:  # GradientBoosting
        param_grid = {
            'n_estimators': [300],
            'learning_rate': [0.1, 0.15],
            'max_depth': [5, 7],
            'subsample': [0.9],
            'min_samples_split': [2, 5]
        }
    
    grid_search = GridSearchCV(
        config['model'], param_grid, cv=5, 
        scoring='r2', n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    # Cross-validation score
    cv_score = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5, scoring='r2').mean()
    print(f"{name} CV R² Score: {cv_score:.4f}")
    
    if cv_score > best_score:
        best_score = cv_score
        best_model = grid_search.best_estimator_
        best_name = name

print(f"\nBest model: {best_name} with CV R² Score: {best_score:.4f}")

# ---------- Step 8: Model Evaluation ----------
y_pred = best_model.predict(X_test)

print("\n" + "="*50)
print("✅ IMPROVED MODEL EVALUATION:")
print("="*50)
print(f"Model: {best_name}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n🔍 Top 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))

# ---------- Step 9: Enhanced Prediction Function ----------
def predict_pm25_enhanced(merra_file, aod_value, target_date="2024-06-10"):
    """Enhanced prediction function with proper preprocessing"""
    
    # Extract MERRA features (simplified for single day)
    ds = xr.open_dataset(merra_file)
    lat, lon = 28.41, 77.31
    
    nearest_lat = ds.sel(lat=lat, method="nearest").lat.values
    nearest_lon = ds.sel(lon=lon, method="nearest").lon.values
    time_step = ds.time.values[0]
    
    # Create base features dictionary
    base_data = {
        "Date": [pd.to_datetime(target_date)],
        "Mean_AOD": [aod_value],
        "PS_mean": [float(ds["PS"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "QV2M_mean": [float(ds["QV2M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "T2M_mean": [float(ds["T2M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "TS_mean": [float(ds["TS"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "U10M_mean": [float(ds["U10M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "QV10M_mean": [float(ds["QV10M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "SLP_mean": [float(ds["SLP"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "T10M_mean": [float(ds["T10M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "T2MDEW_mean": [float(ds["T2MDEW"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "TQI_mean": [float(ds["TQI"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "TQL_mean": [float(ds["TQL"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
    }
    
    # Add dummy std values (you should calculate these from multiple time steps if available)
    for var in ['PS', 'QV2M', 'T2M', 'TS', 'U10M', 'QV10M', 'SLP', 'T10M', 'T2MDEW', 'TQI', 'TQL']:
        base_data[f"{var}_std"] = [0.1]  # Placeholder values
    
    # Create DataFrame and apply feature engineering
    predict_df = pd.DataFrame(base_data)
    predict_df = create_advanced_features(predict_df)
    
    # Select and scale features
    X_predict = predict_df[selected_features].fillna(0)
    X_predict_scaled = scaler.transform(X_predict)
    
    # Make prediction
    prediction = best_model.predict(X_predict_scaled)[0]
    
    return prediction

# ---------- Step 10: Make Prediction ----------
try:
    merra_file = "merra_downloads/MERRA2_400.tavg1_2d_slv_Nx.20240610.SUB.nc"
    predicted_pm = predict_pm25_enhanced(merra_file, aod_value=0.97)
    
    print(f"\n🔮 Enhanced Predicted PM2.5 for June 10, 2024: {predicted_pm:.2f} µg/m³")
    
    # Save prediction
    output_df = pd.DataFrame({
        'Date': ['2024-05-11'],
        'Predicted_PM2.5': [predicted_pm],
        'Model_Used': [best_name],
        'R2_Score': [r2_score(y_test, y_pred)]
    })
    output_df.to_csv('improved_predicted_pm25.csv', index=False)
    print("💾 Enhanced prediction saved to 'improved_predicted_pm25.csv'")
    
except Exception as e:
    print(f"Error in prediction: {e}")

