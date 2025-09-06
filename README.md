# AeroSphinx – Surface-Level PM2.5 Estimation & Forecasting

AeroSphinx is a project developed for ISRO’s Bhartiya Antariksha Hackathon (BAH 25).  
It focuses on estimating and forecasting **surface-level PM2.5 concentrations in Faridabad** using:  

- Satellite data (INSAT AOD)  
- Ground monitoring data (CPCB)  
- Reanalysis data (MERRA-2)  

---

## 🔹 Features
- Data preprocessing and integration from heterogeneous sources  
- Machine learning pipeline for PM2.5 estimation  
- Short-term forecasting models  
- Interactive dashboard for real-time visualization  

---

## 🔹 Tech Stack
- **Python:** Pandas, NumPy, Scikit-learn, Matplotlib  
- **Visualization:** Plotly/Dash or Streamlit  
- **Data Sources:** INSAT AOD, CPCB, MERRA-2  

---

## 🔹 Additional Insights

### 1. Feature Engineering
We engineered **two additional features**:  
- **Temperature Difference**  
- **Humidity Ratio**  

These were introduced to capture more accurate trends of how temperature and humidity influence PM2.5 levels. Rising temperatures often cause water droplets to settle along with pollutants, thereby reducing concentrations. Conversely, higher humidity tends to increase PM2.5 concentrations, making these features critical for improved predictions.

---

### 2. Forecast Horizon
Our forecasting model is currently designed for a **one-day horizon**. Specifically, it predicts PM2.5 concentrations around **5:00 PM each evening**. For this phase of the project, we focused only on short-term forecasting to ensure robust evaluation of our pipeline.

---

### 3. Seasonality Trends
Clear seasonality trends were observed:  
- **Winter (around November):** PM2.5 levels increase significantly.  
- **Monsoon season:** Concentrations reach their minimum.  

These seasonal variations highlight the influence of meteorological conditions on air quality and align with real-world environmental observations.

---
