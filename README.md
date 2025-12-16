# India Air Quality Analysis - CMP7005 PRAC1

**Student ID:** ST20316895  
**Module:** CMP7005 (Programming for Data Analysis)  
**Academic Year:** 2025-26

## Project Overview
This project provides a comprehensive analysis of air quality data across 26 Indian cities from 2015 to 2020. It combines robust data preprocessing, exploratory data analysis (EDA), and machine learning to predict Air Quality Index (AQI) levels. The final output includes a deployment-ready Streamlit dashboard for real-time visualization and prediction.

## Key Features & Methodology

### 1. Data Handling & Preprocessing
- **Merging:** Combined 28 individual city datasets into a single master file (`00_MERGED_Air_Quality_India_2015_2020.csv`).
- **Cleaning:** Standardized column names, parsed dates, and handled missing values using median imputation.
- **Outlier Detection:** Applied IQR-based capping (Winsorization) to handle extreme pollutant values.
- **Scaling:** Standardized numeric features (PM2.5, NO2, etc.) using `StandardScaler` (Z-score normalization) for optimal model performance.
- **Optimization:** Exported the final processed dataset as a compressed GZIP file (`India_Air_Quality_Final_Processed.csv.gz`) to ensure efficient storage and GitHub compatibility (<50MB).

### 2. Exploratory Data Analysis (EDA)
- **Trends:** Visualized AQI trends over time across different cities.
- **Correlations:** Generated heatmaps to identify relationships between major pollutants (PM2.5, PM10, NO2, CO, etc.).
- **Seasonal Patterns:** Analyzed AQI variations across Winter, Summer, Monsoon, and Spring.
- **PCA:** Performed Principal Component Analysis (3D) to inspect data clusters and separability.

### 3. Machine Learning
- **Model:** Trained a **Random Forest Regressor** to predict AQI based on pollutant levels.
- **Evaluation Metrics:**
  - **R² Score:** Measures prediction accuracy.
  - **MAE (Mean Absolute Error):** Average error in AQI points.
  - **MSE (Mean Squared Error):** Penalizes larger errors.
- **Persistence:** Saved the trained model using `joblib` (`aqi_model.pkl`) for live inference in the app.

### 4. Interactive Dashboard (Streamlit)
The project includes a fully functional web app (`app.py`) with the following tabs:
- **Home:** Project KPIs (Total Records, Cities, Avg AQI) and national trends.
- **EDA:** Interactive correlation matrices.
- **Seasonal:** Box plots showing seasonal pollution variances.
- **Model:** Live training interface showing R², MSE, and MAE scores.
- **Predict:** Real-time AQI calculator using Z-score sliders (-5.0 to +5.0) and a severity gauge.
- **Map:** Geospatial visualization of pollution hotspots across India.

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
