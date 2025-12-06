# ==============================================
# AIR QUALITY ANALYSIS DASHBOARD - app.py
# Fully Fixed & Enhanced Version (Dec 2025)
# ==============================================
# ==============================================
# STEP 6: GENERATE STREAMLIT APP FILE (app.py)
# ==============================================

import os

# Define the path where the app.py file will be saved
# You can change this to your desired location in Google Drive
app_file_path = '/content/drive/MyDrive/Programming for Data Analysis/app.py' 

print(f"Creating Streamlit app at: {app_file_path} ...")

# The entire app code as a multi-line string
app_code = """
# ==============================================
# INDIA AIR QUALITY ANALYSIS DASHBOARD
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ==============================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================
st.set_page_config(
    page_title="India Air Quality Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern look
st.markdown(\"\"\"
<style>
    /* Global Background */
    .stApp { background: linear-gradient(to bottom, #f8f9fa, #ffffff); }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem; color: #1E3A8A; text-align: center; padding: 1rem;
        font-weight: 800; background: -webkit-linear-gradient(45deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .section-header {
        font-size: 1.8rem; color: #10B981; margin-top: 1.5rem; margin-bottom: 1rem;
        border-bottom: 2px solid #3B82F6; padding-bottom: 0.5rem; font-weight: 700;
    }
    
    /* Cards/Boxes */
    .metric-card {
        background: white; padding: 1.2rem; border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center;
        border-left: 5px solid #3B82F6;
    }
    .info-box {
        background-color: #EFF6FF; padding: 1rem; border-radius: 10px;
        border-left: 5px solid #3B82F6; margin-bottom: 1rem; color: #1E3A8A;
    }
    .success-box {
        background-color: #ECFDF5; padding: 1rem; border-radius: 10px;
        border-left: 5px solid #10B981; margin-bottom: 1rem; color: #065F46;
    }
    
    /* Interactive Elements */
    .stButton>button {
        width: 100%; border-radius: 8px; font-weight: 600;
        background: linear-gradient(90deg, #3B82F6, #2563EB); border: none;
        color: white; padding: 0.6rem; transition: transform 0.2s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 4px 12px rgba(37,99,235,0.2); }
</style>
\"\"\", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌫️ India Air Quality Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<div class="info-box"><strong>Interactive Analytics Platform</strong> | Explore air quality trends, visualize pollutant distributions, and predict AQI using Machine Learning.</div>', unsafe_allow_html=True)

# ==============================================
# 2. DATA LOADING & CACHING
# ==============================================

@st.cache_data(ttl=3600)
def load_dataset():
    # Attempt to load local/Drive file first
    possible_paths = [
        'preprocessed_air_quality_data.csv',
        '/content/drive/MyDrive/Programming for Data Analysis/preprocessed_air_quality_data.csv',
        'air_quality_data.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # Ensure date column is datetime
                date_cols = [c for c in df.columns if 'date' in c.lower()]
                if date_cols:
                    df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
                    df['date'] = df[date_cols[0]] # Standardize name
                return df
            except: continue
            
    # Fallback: Generate sample data if no file found (for demo purposes)
    dates = pd.date_range(start='2015-01-01', periods=1000, freq='D')
    data = {
        'date': dates,
        'city': np.random.choice(['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bangalore'], 1000),
        'PM2.5': np.random.gamma(2, 20, 1000),
        'PM10': np.random.gamma(3, 30, 1000),
        'NO2': np.random.normal(30, 10, 1000).clip(0),
        'SO2': np.random.normal(15, 5, 1000).clip(0),
        'CO': np.random.normal(1, 0.5, 1000).clip(0),
        'O3': np.random.normal(40, 15, 1000).clip(0),
        'AQI': np.random.gamma(5, 30, 1000) # Derived target
    }
    return pd.DataFrame(data)

@st.cache_resource
def load_ml_model():
    # Attempt to load saved model
    model_paths = [
        'xgboost_aqi_model.pkl',
        '/content/drive/MyDrive/Programming for Data Analysis/xgboost_aqi_model.pkl',
        'best_model.pkl'
    ]
    for path in model_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except: continue
    return None

df = load_dataset()
model = load_ml_model()

# ==============================================
# 3. SIDEBAR CONTROLS
# ==============================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/air-quality.png", width=80)
    st.markdown("## 🎛️ Dashboard Controls")
    
    # Filter Data
    if 'city' in df.columns:
        city_list = ['All'] + sorted(df['city'].unique().tolist())
        selected_city = st.selectbox("Select City", city_list)
    else:
        selected_city = 'All'
        
    # Date Range
    if 'date' in df.columns:
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        date_range = st.date_input("Date Range", [min_date, max_date])
    
    st.markdown("---")
    st.info("Developed for CMP7005\nCardiff Metropolitan University")

# Apply Filters
filtered_df = df.copy()
if selected_city != 'All':
    filtered_df = filtered_df[filtered_df['city'] == selected_city]
if 'date' in filtered_df.columns and len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['date'].dt.date >= date_range[0]) & 
        (filtered_df['date'].dt.date <= date_range[1])
    ]

# ==============================================
# 4. MAIN TABS
# ==============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🔍 Data Explorer", "📈 Deep Dive EDA", "🔮 AQI Predictor", "🗺️ Geospatial"
])

# --- TAB 1: OVERVIEW ---
with tab1:
    st.markdown('<h2 class="section-header">Dashboard Overview</h2>', unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    target_col = 'AQI' if 'AQI' in df.columns else df.select_dtypes(include=np.number).columns[-1]
    
    with col1:
        st.markdown(f'<div class="metric-card"><h3>Records</h3><h2>{len(filtered_df):,}</h2></div>', unsafe_allow_html=True)
    with col2:
        avg_val = filtered_df[target_col].mean()
        st.markdown(f'<div class="metric-card"><h3>Avg {target_col}</h3><h2>{avg_val:.1f}</h2></div>', unsafe_allow_html=True)
    with col3:
        max_val = filtered_df[target_col].max()
        st.markdown(f'<div class="metric-card"><h3>Max {target_col}</h3><h2>{max_val:.1f}</h2></div>', unsafe_allow_html=True)
    with col4:
        cities_count = filtered_df['city'].nunique() if 'city' in df.columns else 1
        st.markdown(f'<div class="metric-card"><h3>Cities</h3><h2>{cities_count}</h2></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # Overview Charts
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📈 Trend Over Time")
        if 'date' in filtered_df.columns:
            # Resample for cleaner chart if too many points
            chart_df = filtered_df.set_index('date').resample('M')[target_col].mean().reset_index()
            fig = px.line(chart_df, x='date', y=target_col, markers=True, line_shape='spline',
                          color_discrete_sequence=['#3B82F6'])
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No date column found for time series.")
            
    with c2:
        st.subheader("📉 Distribution")
        fig = px.histogram(filtered_df, x=target_col, nbins=30, color_discrete_sequence=['#10B981'])
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: DATA EXPLORER ---
with tab2:
    st.markdown('<h2 class="section-header">Data Explorer</h2>', unsafe_allow_html=True)
    
    # Interactive Table
    with st.expander("Filter & Sort Options", expanded=True):
        c1, c2 = st.columns(2)
        sort_col = c1.selectbox("Sort By", filtered_df.columns)
        sort_asc = c2.radio("Order", ["Ascending", "Descending"]) == "Ascending"
    
    st.dataframe(
        filtered_df.sort_values(sort_col, ascending=sort_asc),
        use_container_width=True,
        height=500
    )

# --- TAB 3: DEEP DIVE EDA ---
with tab3:
    st.markdown('<h2 class="section-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    type_ = st.selectbox("Analysis Type", ["Correlation Heatmap", "Bivariate Scatter", "Box Plots"])
    
    if type_ == "Correlation Heatmap":
        num_df = filtered_df.select_dtypes(include=np.number)
        if len(num_df.columns) > 1:
            corr = num_df.corr()
            fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', aspect='auto')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numeric columns for correlation.")
            
    elif type_ == "Bivariate Scatter":
        num_cols = filtered_df.select_dtypes(include=np.number).columns
        c1, c2 = st.columns(2)
        x_axis = c1.selectbox("X Axis", num_cols, index=0)
        y_axis = c2.selectbox("Y Axis", num_cols, index=min(1, len(num_cols)-1))
        
        fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color=target_col, 
                         color_continuous_scale='Viridis', opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
        
    elif type_ == "Box Plots":
        num_cols = filtered_df.select_dtypes(include=np.number).columns
        y_col = st.selectbox("Select Variable", num_cols)
        # Group by Year or Month if available, else City
        if 'city' in filtered_df.columns:
            x_col = 'city'
        else:
            x_col = None
            
        fig = px.box(filtered_df, x=x_col, y=y_col, color=x_col)
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 4: PREDICTOR ---
with tab4:
    st.markdown('<h2 class="section-header">🔮 AQI Predictor</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Input Parameters")
        input_data = {}
        # Dynamic inputs based on numeric columns (excluding target/dates)
        feature_cols = [c for c in df.select_dtypes(include=np.number).columns 
                        if c not in [target_col, 'year', 'month', 'day']]
        
        for col in feature_cols[:6]: # Limit to top 6 features for UI cleanliness
            min_v = float(df[col].min())
            max_v = float(df[col].max())
            mean_v = float(df[col].mean())
            input_data[col] = st.slider(f"{col}", min_v, max_v, mean_v)
            
        predict_btn = st.button("Predict AQI", type="primary")

    with col2:
        st.markdown("### Prediction Result")
        if predict_btn:
            if model:
                try:
                    # Prepare input dataframe
                    input_df = pd.DataFrame([input_data])
                    # Add dummy values for missing columns expected by model
                    # (In a real scenario, you'd match the exact training features)
                    # Here we assume the model uses the features we collected
                    
                    pred = model.predict(input_df)[0]
                    
                    # Display Result
                    st.markdown(f'''
                    <div style="text-align: center; padding: 2rem; border-radius: 15px; 
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                        <h1 style="font-size: 4rem; margin: 0;">{pred:.0f}</h1>
                        <h3>Predicted {target_col}</h3>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Context
                    if pred <= 50: status, color = "Good", "green"
                    elif pred <= 100: status, color = "Moderate", "gold"
                    elif pred <= 200: status, color = "Poor", "orange"
                    else: status, color = "Severe", "red"
                    
                    st.markdown(f"### Status: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    st.info("Ensure the input features match the trained model.")
            else:
                st.warning("⚠️ No machine learning model loaded. Please train/save a model first.")
                # Fallback dummy prediction for demo
                st.info("Demonstration Prediction (Random):")
                st.metric("Predicted AQI", f"{np.random.randint(50, 300)}")

# --- TAB 5: GEOSPATIAL ---
with tab5:
    st.markdown('<h2 class="section-header">Geospatial Analysis</h2>', unsafe_allow_html=True)
    
    if 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
        fig = px.scatter_mapbox(filtered_df, lat="latitude", lon="longitude", color=target_col,
                                size=target_col, color_continuous_scale=px.colors.cyclical.IceFire,
                                zoom=3, mapbox_style="open-street-map",
                                hover_name="city" if 'city' in filtered_df.columns else None)
        fig.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Latitude/Longitude columns found in the dataset for map visualization.")
        st.write("Using simulated map for demonstration:")
        
        # Simulated Map Data
        sim_data = pd.DataFrame({
            'lat': [28.61, 19.07, 13.08, 22.57, 12.97],
            'lon': [77.20, 72.87, 80.27, 88.36, 77.59],
            'city': ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bangalore'],
            'AQI': [300, 150, 120, 200, 90]
        })
        fig = px.scatter_mapbox(sim_data, lat="lat", lon="lon", color="AQI", size="AQI",
                                zoom=3, mapbox_style="open-street-map", hover_name="city")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>© 2025 India Air Quality Project | Built with Streamlit</div>", unsafe_allow_html=True)
"""

# Write the file
with open(app_file_path, "w") as f:
    f.write(app_code)

print("✅ App file created successfully!")
print(f"File location: {app_file_path}")
print("\nTo run this app locally:")
print(f"1. Download '{os.path.basename(app_file_path)}' and your data/model files.")
print("2. Open terminal/cmd.")
print("3. Run: streamlit run app.py")
