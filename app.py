# ============================================================================
# INDIA AIR QUALITY ANALYSIS APP (CMP7005 ASSESSMENT)
# DEVELOPED BY: MD RABIUL ALAM
# STUDENT ID: ST20316895
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import warnings

# Suppress warnings for a cleaner UI
warnings.filterwarnings("ignore")

# ============================================================================
# 1. APP CONFIGURATION
# ============================================================================
st.set_page_config(page_title="India Air Quality", layout="wide", page_icon="🌤️")

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        background: linear-gradient(90deg, #001f3f, #003366); 
        padding: 30px; border-radius: 15px; color: white; 
        text-align: center; margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .title { font-size: 50px; font-weight: bold; margin: 0; }
    .subtitle { font-size: 24px; margin-top: 10px; color: #e2e8f0; }
    .metric-card {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        padding: 20px; border-radius: 12px; color: white;
        text-align: center; border: 1px solid #334155;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card h2 { margin: 10px 0 0 0; color: #38bdf8; }
    .stTabs [data-testid="stTab"] { font-weight: bold; font-size: 16px; }
</style>
<div class="header">
    <div class="title">India Air Quality Analysis</div>
    <div class="subtitle">CMP7005 • ST20316895 • 2025-26</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 2. DATA LOADING (WITH SMART RECOVERY)
# ============================================================================
@st.cache_data
def load_data():
    try:
        # 1. Load the Processed (Scaled) Data
        df = pd.read_csv("India_Air_Quality_Final_Processed.csv.gz", compression='gzip')
        
        # Clean column names (remove spaces)
        df.columns = [c.strip() for c in df.columns]

        # 2. Rename columns to standard format (handles case sensitivity)
        rename_map = {'city': 'City', 'date': 'Date', 'aqi': 'AQI'}
        new_names = {c: rename_map[c.lower()] for c in df.columns if c.lower() in rename_map}
        df.rename(columns=new_names, inplace=True)

        # 3. SMART RECOVERY (The Fix for "1 City" Issue)
        # If 'City' is missing or has only 1 unique value, we attempt to merge 
        # city names from the original raw file (Task 1 output).
        if 'City' not in df.columns or df['City'].nunique() <= 1:
            try:
                # Load the original raw file that definitely has all cities
                raw_df = pd.read_csv("00_MERGED_Air_Quality_India_2015_2020.csv")
                
                # If row counts are compatible, restore the City and Date columns
                # We use the length of the smaller dataframe to avoid index errors
                limit = min(len(df), len(raw_df))
                
                if 'City' in raw_df.columns:
                    df.loc[:limit-1, 'City'] = raw_df['City'].iloc[:limit].values
                
                if 'Date' not in df.columns and 'Date' in raw_df.columns:
                    df.loc[:limit-1, 'Date'] = raw_df['Date'].iloc[:limit].values
                    
            except FileNotFoundError:
                # If raw file is missing, we can't fix it automatically
                pass 

        # 4. Final Cleanup & Type Conversion
        if 'Date' in df.columns: 
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        if 'AQI' in df.columns: 
            df = df.dropna(subset=['AQI'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load Data
df = load_data()

# Stop app if data failed to load
if df.empty:
    st.error("Data could not be loaded. Please ensure 'India_Air_Quality_Final_Processed.csv' is in the directory.")
    st.stop()

# ============================================================================
# 3. VISUALIZATION TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Home", "EDA", "Seasonal", "Model", "Predict", "Map", "About"])

# --- TAB 1: DASHBOARD ---
with tab1:
    st.header("Project Dashboard")
    
    # Calculate KPIs
    # Use robust checks in case 'City' is still missing
    total_cities = df['City'].nunique() if 'City' in df.columns else 0
    avg_aqi = df['AQI'].mean() if 'AQI' in df.columns else 0
    max_aqi = df['AQI'].max() if 'AQI' in df.columns else 0

    # Display KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card">Records<br><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card">Cities<br><h2>{total_cities}</h2></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card">Avg AQI<br><h2>{avg_aqi:.1f}</h2></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card">Max AQI<br><h2>{max_aqi:.0f}</h2></div>', unsafe_allow_html=True)

    # Trend Chart
    if 'Date' in df.columns and 'City' in df.columns:
        st.markdown("### National AQI Trends")
        # Sample data for performance if dataset is large
        plot_data = df.sample(min(5000, len(df))) if len(df) > 5000 else df
        fig = px.line(plot_data, x='Date', y='AQI', color='City', title="AQI Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ 'Date' or 'City' column missing. Trends cannot be displayed.")

# --- TAB 2: EDA ---
with tab2:
    st.header("Correlation Analysis")
    
    # Select only numeric columns for correlation
    numeric = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
    
    if not numeric.empty:
        # Adaptive Heatmap (Height increased for visibility)
        fig = px.imshow(numeric.corr(), text_auto=True, color_continuous_scale='RdBu_r', height=700, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough numeric data for correlation.")

# --- TAB 3: SEASONAL ---
with tab3:
    st.header("Seasonal Patterns")
    
    if 'Date' in df.columns and 'AQI' in df.columns:
        # Create seasonal mapping
        df['Season'] = df['Date'].dt.month.map({
            12:'Winter', 1:'Winter', 2:'Winter', 
            3:'Spring', 4:'Spring', 5:'Spring', 
            6:'Summer', 7:'Summer', 8:'Summer', 
            9:'Monsoon', 10:'Monsoon', 11:'Monsoon'
        })
        
        fig = px.box(df, x='Season', y='AQI', color='Season', 
                     title="Seasonal Air Quality Levels", 
                     category_orders={"Season": ["Winter", "Spring", "Summer", "Monsoon"]})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ 'Date' column missing (Check CSV). Cannot show seasonal analysis.")

# --- TAB 4: MODEL ---
with tab4:
    st.header("Model Training")
    st.markdown("Training Random Forest on **Scaled Data**.")
    
    if st.button("🚀 Train Model Now", type="primary"):
        with st.spinner("Training..."):
            # Prepare Data (Dropping Target & Non-Numeric)
            X = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
            y = df['AQI']
            
            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize & Train Model
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Save Model
            joblib.dump(model, "aqi_model.pkl")
            
            # Show Result
            score = r2_score(y_test, model.predict(X_test))
            st.success(f"Model Trained Successfully! R² Score: {score:.4f}")
            st.balloons()

# --- TAB 5: PREDICT (SCALED) ---
with tab5:
    st.header("Predict Air Quality")
    
    try:
        model = joblib.load("aqi_model.pkl")
        
        # Get feature names from dataframe
        X_feats = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
        
        st.info("ℹ️ **Input: Standardized Z-Scores (-5.0 to +5.0)**\n0.0 = Average, +2.0 = High, -2.0 = Low")
        
        inputs = []
        cols = st.columns(3)
        
        # Show sliders for top 9 features
        for i, f in enumerate(X_feats.columns[:9]): 
            with cols[i % 3]:
                inputs.append(st.slider(f"{f} (Z-Score)", -5.0, 5.0, 0.0, 0.1))
        
        if st.button("Predict AQI", type="primary", use_container_width=True):
            # Pad remaining features with 0.0 (Average)
            full_in = inputs + [0.0] * (len(X_feats.columns) - len(inputs))
            
            # Predict
            pred = model.predict([full_in])[0]
            
            # Display
            st.divider()
            c1, c2 = st.columns([1,2])
            with c1: 
                st.markdown("### Prediction")
                st.markdown(f"<h1 style='color:#10b981;font-size:60px;margin:0'>{pred:.0f}</h1>", unsafe_allow_html=True)
            with c2: 
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", 
                    value=pred, 
                    title={'text': "Severity"},
                    gauge={'axis': {'range': [0, 500]}, 'bar': {'color': "white"}, 
                           'steps': [{'range': [0,100], 'color': "#00b894"}, 
                                     {'range': [100,200], 'color': "#fdcb6e"}, 
                                     {'range': [200,500], 'color': "#d63031"}]}
                ))
                fig.update_layout(height=250, margin=dict(t=30,b=20,l=20,r=20))
                st.plotly_chart(fig, use_container_width=True)
                
    except FileNotFoundError:
        st.warning("⚠️ Model not found. Please go to the 'Model' tab and click 'Train Model Now'.")

# --- TAB 6: MAP ---
with tab6:
    st.header("Pollution Hotspots")
    
    if 'City' in df.columns and 'AQI' in df.columns:
        # Manually defined coordinates for major Indian cities
        coords = {
            'Delhi':(28.61,77.20), 'Mumbai':(19.07,72.87), 'Bengaluru':(12.97,77.59), 
            'Kolkata':(22.57, 88.36), 'Chennai':(13.08, 80.27), 'Hyderabad':(17.38, 78.48), 
            'Ahmedabad':(23.02, 72.57), 'Lucknow':(26.84, 80.94), 'Patna':(25.59, 85.13), 
            'Gurugram':(28.45, 77.02), 'Amritsar':(31.63, 74.87), 'Jaipur':(26.91, 75.78), 
            'Visakhapatnam':(17.68, 83.21), 'Thiruvananthapuram':(8.52, 76.93), 
            'Nagpur':(21.14, 79.08), 'Chandigarh':(30.73, 76.77), 'Bhopal':(23.25, 77.41), 
            'Shillong':(25.57, 91.89)
        }
        
        # Aggregate AQI by City
        city_stats = df.groupby('City')['AQI'].mean().reset_index()
        
        # Map lat/lon
        city_stats['lat'] = city_stats['City'].map(lambda x: coords.get(x, (None,None))[0])
        city_stats['lon'] = city_stats['City'].map(lambda x: coords.get(x, (None,None))[1])
        
        # Plot Map
        fig = px.scatter_mapbox(city_stats.dropna(subset=['lat']), lat="lat", lon="lon", 
                                size="AQI", color="AQI", hover_name="City", zoom=3.5, 
                                color_continuous_scale="RdYlGn_r", title="Average AQI by City")
        fig.update_layout(mapbox_style="carto-positron", height=600)
        st.plotly_chart(fig, use_container_width=True)
    else: 
        st.warning("⚠️ 'City' column missing (Check CSV). Cannot render map.")

# --- TAB 7: ABOUT ---
with tab7:
    st.header("About")
    st.markdown("""
    ### India Air Quality Analysis App
    **CMP7005 Data Analysis Assessment**
    
    * **Student ID:** ST20316895
    * **Name:** MD RABIUL ALAM
    * **Module Leader:** aprasad@cardiffmet.ac.uk
    
    **Features:**
    * **Data Processing:** Handles standardized/scaled pollutant data.
    * **Visualization:** Interactive trends, correlations, and maps using Plotly.
    * **Machine Learning:** Random Forest Regressor trained live on the processed dataset.
    * **Prediction:** Real-time AQI inference based on pollutant Z-scores.
    """)
