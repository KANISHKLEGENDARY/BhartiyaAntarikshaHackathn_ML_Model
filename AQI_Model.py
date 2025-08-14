#Combining MOSDAC and CPCB data with MERRA and predicting the PM2.5 Level(SAutomatic filling of MERRA DATA)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from netCDF4 import Dataset, num2date
from datetime import datetime
import os
import xarray as xr

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

# ---------- Step 3: Load MERRA .nc4 files ----------
def extract_merra_features(nc_folder):
    records = []
    for file in os.listdir(nc_folder):
        if file.endswith(".nc"):
            ds = Dataset(os.path.join(nc_folder, file), 'r')

            time_var = ds.variables['time']
            times = num2date(time_var[:], units=time_var.units, only_use_cftime_datetimes=False)

            ps = ds.variables['PS'][:, 0, 0]
            qv2m = ds.variables['QV2M'][:, 0, 0]
            t2m = ds.variables['T2M'][:, 0, 0]
            ts = ds.variables['TS'][:, 0, 0]
            u10m = ds.variables['U10M'][:, 0, 0]
            # v10m = ds.variables['V10M'][:, 0, 0]
            qv10m = ds.variables['QV10M'][:, 0, 0]
            slp = ds.variables['SLP'][:, 0, 0]
            t10m = ds.variables['T10M'][:, 0, 0]
            t2mdew = ds.variables['T2MDEW'][:, 0, 0]
            tqi = ds.variables['TQI'][:, 0, 0]
            tql = ds.variables['TQL'][:, 0, 0]

            for i in range(len(times)):
                # Ensure times[i] is a native datetime object
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
                    # "U2M": u2m[i]
                })

    return pd.DataFrame(records)

merra_df = extract_merra_features("merra_downloads")  
merra_df = merra_df.groupby("Date").mean().reset_index()

aod_df['Date'] = pd.to_datetime(aod_df['Date'])
pm_df['Date'] = pd.to_datetime(pm_df['Date'])
merra_df['Date'] = pd.to_datetime(merra_df['Date'])  

# print(aod_df.dtypes)
# print(pm_df.dtypes)
# print(merra_df.dtypes)

# print(f"AOD records: {len(aod_df)}")
# print(f"CPCB records: {len(pm_df)}")
# print(f"MERRA records: {len(merra_df)}")

# ---------- Step 4: Merge All ----------
combined_df = pd.merge(aod_df, pm_df, on="Date")
# print(f"After merging AOD + CPCB: {len(combined_df)} records")

combined_df = pd.merge(combined_df, merra_df, on="Date")
# print(f"After merging with MERRA: {len(combined_df)} records")

# print("🔍 AOD dates:", aod_df['Date'].unique())
# print("🔍 CPCB dates:", pm_df['Date'].unique())
# print("🔍 MERRA dates:", merra_df['Date'].unique())


# ---------- Step 5: ML Model ----------
features = ['Mean_AOD', 'PS', 'QV2M', 'T2M', 'TS', 'U10M', 'QV10M', 'SLP', 'T10M', 'T2MDEW', 'TQI', 'TQL']
clean_df = combined_df.dropna(subset=features + ['PM2.5 (µg/m³)'])

# Create feature matrix and target vector
X = clean_df[features].copy()
y = clean_df['PM2.5 (µg/m³)']

# Optional feature engineering
X['Temp_Diff'] = X['TS'] - X['T2M']
X['Humidity_Ratio'] = X['QV2M'] / (X['T2M'] + 1e-3)  # prevent division by zero

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("✅ Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))


# ---------- Step 6: Extract MERRA features for a specific date ----------
def extract_merra_single_day(file_path, lat=28.41, lon=77.31):
    ds = xr.open_dataset(file_path)

    nearest_lat = ds.sel(lat=lat, method="nearest").lat.values
    nearest_lon = ds.sel(lon=lon, method="nearest").lon.values

    time_step = ds.time.values[0]  # Assuming only one day present

    features = {
        "PS":     [float(ds["PS"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "QV2M":   [float(ds["QV2M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "T2M":    [float(ds["T2M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "TS":     [float(ds["TS"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "U10M":   [float(ds["U10M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "QV10M":  [float(ds["QV10M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "SLP":    [float(ds["SLP"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "T10M":   [float(ds["T10M"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "T2MDEW": [float(ds["T2MDEW"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "TQI":    [float(ds["TQI"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
        "TQL":    [float(ds["TQL"].sel(time=time_step, lat=nearest_lat, lon=nearest_lon).values)],
    }

    return features

# Update this path to the correct file for the target date
merra_file = "merra_downloads/MERRA2_400.tavg1_2d_slv_Nx.20240403.SUB.nc"
merra_features = extract_merra_single_day(merra_file)

# Add manually or automate fetching AOD from INSAT later
merra_features["Mean_AOD"] = [0.97]

# Create DataFrame from extracted features
predict_input_df = pd.DataFrame(merra_features)

# ---------- Step 7: Feature Engineering ----------
predict_input_df['Temp_Diff'] = predict_input_df['TS'] - predict_input_df['T2M']
predict_input_df['Humidity_Ratio'] = predict_input_df['QV2M'] / (predict_input_df['T2M'] + 1e-3)

# Ensure correct column order
X_input = predict_input_df[X_train.columns]

# ---------- Step 8: Predict ----------
pred_pm = rf.predict(X_input)
print("\n🔮 Predicted PM2.5 for 03-04-2024 around 05:30 IST:", pred_pm[0])

# ---------- Step 9: Save prediction ----------
output_df = pd.DataFrame({
    'Date': ['2024-05-11'],
    'Predicted_PM2.5': [pred_pm[0]]
})
output_df.to_csv('predicted_pm25.csv', index=False, mode='w')
print("💾 Prediction saved to 'predicted_pm25.csv'")
