import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import os
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# After defining filters, replace the filtering block with:
filtered_df = df.copy()

# Safe date filtering
if 'date' in df.columns and 'date_range' in locals() and len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df['date'].dt.date >= start_date) & 
        (filtered_df['date'].dt.date <= end_date)
    ]

# Safe city filtering
if 'selected_cities' in locals() and selected_cities and 'All' not in selected_cities:
    filtered_df = filtered_df[filtered_df['city'].isin(selected_cities)]

# Safe pollutant range filtering
if 'value_range' in locals() and pollutant in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df[pollutant] >= value_range[0]) & 
        (filtered_df[pollutant] <= value_range[1])
    ]

if uploaded_file is not None:
    # ... reading logic ...
    st.session_state.uploaded_df = df  # Persist uploaded data
    st.success(f"Uploaded: {uploaded_file.name} ({len(df)} rows)")

# Then in load_data():
if 'uploaded_df' in st.session_state:
    return st.session_state.uploaded_df

# Get expected features from config if available
expected_features = config.get('feature_names', ['PM2_5','PM10','NO2','SO2','O3','CO']) if config else None

input_data = pd.DataFrame([{
    'PM2_5': pm25, 'PM10': pm10, 'NO2': no2, 'SO2': so2,
    'O3': o3, 'CO': co,
    'temperature': temperature, 'humidity': humidity, 'wind_speed': wind_speed
    # Add city one-hot if needed later
}])

# Reindex to match training features (if known)
if expected_features:
    input_data = input_data.reindex(columns=expected_features, fill_value=0)
