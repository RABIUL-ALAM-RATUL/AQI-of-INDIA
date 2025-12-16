# India Air Quality Analysis - CMP7005 PRAC1

**Student ID:** ST20316895  
**Module:** CMP7005 (Programming for Data Analysis)  
**Academic Year:** 2025-26

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/blob/main/WRT1_ST20316895_CMP7005_S1_PRAC1_2025_26.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aqi-of-india-md-rabiul-alam-st20316895-wrt1.streamlit.app/)

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
## 🎯 Project Goal
The primary goal of this project is to analyze historical air quality data from 2015 to 2020 across major Indian cities to understand pollution trends, seasonal variations, and key pollutant correlations. By leveraging machine learning, the project aims to build a predictive model capable of estimating the Air Quality Index (AQI) based on specific pollutant concentrations (e.g., PM2.5, NO2, SO2). The final deliverable is an interactive, user-friendly dashboard that empowers stakeholders to visualize these insights and perform real-time AQI predictions.

## 🔍 Project Scope
This project covers the complete data science lifecycle, including:
1.  **Data Ingestion & Cleaning:** Merging 28 city-specific datasets, handling missing values, and standardizing diverse data formats.
2.  **Statistical Analysis:** Conducting rigorous Exploratory Data Analysis (EDA) to uncover temporal trends and pollutant relationships.
3.  **Feature Engineering:** Implementing IQR-based outlier detection, seasonal mapping, and Z-score standardization.
4.  **Predictive Modeling:** Developing and evaluating a Random Forest Regressor to predict AQI values with high accuracy.
5.  **Deployment:** creating a deployed web application (Streamlit) that serves the model and visualizations to end-users.
