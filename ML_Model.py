#Combining MOSDAC and CPCB data with MERRA and predicting the PM2.5 Level(SAutomatic filling of MERRA DATA)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import xarray as xr

# ---------- Step 1: Load INSAT AOD from CSV ----------
aod_df = pd.read_csv("aod_data.csv")
aod_df['Date'] = pd.to_datetime(aod_df['Date'], format='mixed', dayfirst=True, errors='coerce')

# ---------- Step 2: Load CPCB PM2.5 ----------
pm_df = pd.read_csv("combined_pm2.5_para.csv")
pm_df['Timestamp'] = pd.to_datetime(pm_df['Timestamp'], format='mixed', dayfirst=True, errors='coerce')
pm_df['Date'] = pm_df['Timestamp'].dt.normalize()

# ---------- Step 3: Load MERRA .nc4 files ----------
def extract_merra_features(nc_folder, target_lat=28.41, target_lon=77.31):
    records = []
    for file in os.listdir(nc_folder):
        if file.endswith(".nc"):
            ds = xr.open_dataset(os.path.join(nc_folder, file))
            nearest_lat = float(ds.lat.sel(lat=target_lat, method="nearest"))
            nearest_lon = float(ds.lon.sel(lon=target_lon, method="nearest"))
            
            for time_step in ds.time:
                records.append({
                    "Date": pd.to_datetime(str(time_step.values)),
                    "PS": float(ds["PS"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values),
                    "QV2M": float(ds["QV2M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values),
                    "T2M": float(ds["T2M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values),
                    "TS": float(ds["TS"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values),
                    "U10M": float(ds["U10M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values),
                    "QV10M": float(ds["QV10M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values),
                    "SLP": float(ds["SLP"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values),
                    "T10M": float(ds["T10M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values),
                    "T2MDEW": float(ds["T2MDEW"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values),
                    "TQI": float(ds["TQI"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values),
                    "TQL": float(ds["TQL"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values),
                    "PBLTOP": float(ds["PBLTOP"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values),
                    "V10M": float(ds["V10M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values),
                    "V2M": float(ds["V2M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values),
                    "U2M": float(ds["U2M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values),
                })
    return pd.DataFrame(records)

merra_df = extract_merra_features("merra_downloads")  
merra_df = merra_df.groupby("Date").mean().reset_index()

aod_df['Date'] = pd.to_datetime(aod_df['Date'])
pm_df['Date'] = pd.to_datetime(pm_df['Date'])
merra_df['Date'] = pd.to_datetime(merra_df['Date'], utc=True)

# Resample to daily mean
merra_daily = merra_df.set_index('Date').resample('1D').mean().reset_index()

# ---------- Step 4: Merge All ----------
combined_df = pd.merge(aod_df, pm_df, on="Date")
combined_df['Date'] = combined_df['Date'].dt.tz_localize("UTC")
combined_df = pd.merge(combined_df, merra_daily, on="Date")

# ---------- Step 5: ML Model with Hyperparameter Tuning ----------
features = ['Mean_AOD', 'PS', 'QV2M', 'T2M', 'TS', 'U10M', 'QV10M', 'SLP', 'T10M', 'T2MDEW', 'TQI', 'TQL', 'PBLTOP', 'U2M', 'V10M', 'V2M']
clean_df = combined_df.dropna(subset=features + ['PM2.5 (µg/m³)'])

X = clean_df[features].copy()
y = clean_df['PM2.5 (µg/m³)']

# Feature engineering
X['Temp_Diff'] = X['TS'] - X['T2M']
X['Humidity_Ratio'] = X['QV2M'] / (X['T2M'] + 1e-3)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 5: Hyperparameter tuning with more parameters
param_grid = {
    "n_estimators": [200, 400, 600],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)


# ---------- Step 6: Classification Metrics ----------
# Define pollution threshold
threshold = 60
y_pred = best_rf.predict(X_test)
y_test_class = (y_test > threshold).astype(int)
y_pred_class = (y_pred > threshold).astype(int)

print("\n📊 Classification Evaluation (Threshold = 60 µg/m³):")
print("Accuracy:", accuracy_score(y_test_class, y_pred_class))
print("Precision:", precision_score(y_test_class, y_pred_class, zero_division=0))
print("Recall:", recall_score(y_test_class, y_pred_class, zero_division=0))
print("F1 Score:", f1_score(y_test_class, y_pred_class, zero_division=0))

# ---------- Step 7: Predict a Specific Day ----------
def extract_merra_single_day(file_path, lat=28.41, lon=77.31):
    ds = xr.open_dataset(file_path)
    nearest_lat = ds.sel(lat=lat, method="nearest").lat.values
    nearest_lon = ds.sel(lon=lon, method="nearest").lon.values
    time_step = ds.time.values[0]
    features = {
        "PS": [float(ds["PS"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "QV2M": [float(ds["QV2M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "T2M": [float(ds["T2M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "TS": [float(ds["TS"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "U10M": [float(ds["U10M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "QV10M": [float(ds["QV10M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "SLP": [float(ds["SLP"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "T10M": [float(ds["T10M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "T2MDEW": [float(ds["T2MDEW"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "TQI": [float(ds["TQI"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "TQL": [float(ds["TQL"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "PBLTOP": [float(ds["PBLTOP"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "U2M": [float(ds["U2M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "V10M": [float(ds["V10M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "V2M": [float(ds["V2M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
    }
    return features

merra_file = "merra_downloads/MERRA2_400.tavg1_2d_slv_Nx.20240403.SUB.nc"
merra_features = extract_merra_single_day(merra_file)
merra_features["Mean_AOD"] = [0.97]

predict_input_df = pd.DataFrame(merra_features)
predict_input_df['Temp_Diff'] = predict_input_df['TS'] - predict_input_df['T2M']
predict_input_df['Humidity_Ratio'] = predict_input_df['QV2M'] / (predict_input_df['T2M'] + 1e-3)

X_input = predict_input_df[X_train.columns]
pred_pm = best_rf.predict(X_input)

print("\n🔮 Predicted PM2.5 for 03-04-2024 around 05:30 IST:", pred_pm[0])

output_df = pd.DataFrame({'Date': ['2024-05-11'], 'Predicted_PM2.5': [pred_pm[0]]})
output_df.to_csv('predicted_pm25.csv', index=False, mode='w')
print("💾 Prediction saved to 'predicted_pm25.csv'")
