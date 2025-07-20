#Combining MOSDAC and CPCB data with MERRA and predicting the PM2.5 Level
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
import os
import h5py

# ---------- Step 1: Load INSAT AOD from CSV ----------
aod_df = pd.read_csv("aod_data.csv")
aod_df['Date'] = pd.to_datetime(aod_df['Date'])  # 🔄 Convert to datetime for merging
# print(aod_df.head())

# ---------- Step 2: Load CPCB PM2.5 ----------
pm_df = pd.read_csv("pm25_cpcb_05AM_01janto18june.csv")
pm_df['Timestamp'] = pd.to_datetime(pm_df['Timestamp'])
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

merra_df = extract_merra_features("merra_downloads")  # 📁 Folder containing .nc4 files
merra_df = merra_df.groupby("Date").mean().reset_index()

aod_df['Date'] = pd.to_datetime(aod_df['Date'])
pm_df['Date'] = pd.to_datetime(pm_df['Date'])
merra_df['Date'] = pd.to_datetime(merra_df['Date'])  # ✅ This fixes the issue

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

# # ---------- Step 6: Predict for 20 June ----------
# # Replace with actual June 20 values from your MERRA and AOD
may_11 = {
    "Mean_AOD": [0.97],
    "PS": [98119.390],
    "QV2M": [0.007],
    "T2M": [313.68],
    "TS": [316.82],
    "U10M": [2.75],
    "QV10M": [0.007],
    "SLP": [100396.60],
    "T10M": [312.85],
    "T2MDEW": [283.28],
    "TQI": [0.0],
    "TQL": [0.0],
}
may_11_df = pd.DataFrame(may_11)

# Recreate engineered features
may_11_df['Temp_Diff'] = may_11_df['TS'] - may_11_df['T2M']
may_11_df['Humidity_Ratio'] = may_11_df['QV2M'] / (may_11_df['T2M'] + 1e-3)

# Ensure columns match training
X_input = may_11_df[X_train.columns]

# Predict
pred_pm = rf.predict(X_input)
print("\n🔮 Predicted PM2.5 for 11 May 2024 at 05:30 IST:", pred_pm[0])
