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

# Suppress warnings
warnings.filterwarnings("ignore")

# ============================================================================
# 1. APP CONFIGURATION
# ============================================================================
st.set_page_config(page_title="India Air Quality", layout="wide", page_icon="🌤️")

# Custom CSS
st.markdown("""
<style>
    .header {
        background: linear-gradient(90deg, #001f3f, #003366); 
        padding: 30px; border-radius: 15px; color: white; 
        text-align: center; margin-bottom: 30px;
    }
    .title { font-size: 50px; font-weight: bold; margin: 0; }
    .subtitle { font-size: 24px; margin-top: 10px; color: #e2e8f0; }
    .metric-card {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        padding: 20px; border-radius: 12px; color: white;
        text-align: center; border: 1px solid #334155;
    }
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
        df = pd.read_csv("India_Air_Quality_Final_Processed.csv")
        df.columns = [c.strip() for c in df.columns]

        # 2. Rename columns to standard format
        rename_map = {'city': 'City', 'date': 'Date', 'aqi': 'AQI'}
        new_names = {c: rename_map[c.lower()] for c in df.columns if c.lower() in rename_map}
        df.rename(columns=new_names, inplace=True)

        # 3. SMART RECOVERY (The Fix for "1 City")
        # If the processed file lost its City names (shows only 1 city), 
        # we try to grab them from the original merged file.
        if 'City' not in df.columns or df['City'].nunique() <= 1:
            try:
                # Load the original raw file (Task 1 output)
                raw_df = pd.read_csv("00_MERGED_Air_Quality_India_2015_2020.csv")
                
                # If row counts are compatible, restore the City and Date columns
                if len(raw_df) >= len(df):
                    df['City'] = raw_df['City'].iloc[:len(df)].values
                    if 'Date' not in df.columns:
                        df['Date'] = raw_df['Date'].iloc[:len(df)].values
            except:
                pass # If raw file missing, continue as is

        # 4. Final Cleanup
        if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if 'AQI' in df.columns: df = df.dropna(subset=['AQI'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty: st.stop()

# ============================================================================
# 3. TABS & VISUALIZATIONS
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Home", "EDA", "Seasonal", "Model", "Predict", "Map", "About"])

# --- TAB 1: DASHBOARD ---
with tab1:
    st.header("Project Dashboard")
    # KPIs
    total_cities = df['City'].nunique() if 'City' in df.columns else 0
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card">Records<br><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card">Cities<br><h2>{total_cities}</h2></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card">Avg AQI<br><h2>{df["AQI"].mean():.1f}</h2></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card">Max AQI<br><h2>{df["AQI"].max():.0f}</h2></div>', unsafe_allow_html=True)

    # Trend Chart
    if 'Date' in df.columns and 'City' in df.columns:
        # Sample data for performance
        fig = px.line(df.sample(min(5000, len(df))), x='Date', y='AQI', color='City', title="National AQI Trends")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: EDA ---
with tab2:
    st.header("Correlation Analysis")
    numeric = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
    if not numeric.empty:
        # Adaptive Heatmap
        fig = px.imshow(numeric.corr(), text_auto=True, color_continuous_scale='RdBu_r', height=800, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: SEASONAL ---
with tab3:
    st.header("Seasonal Patterns")
    if 'Date' in df.columns:
        df['Season'] = df['Date'].dt.month.map({12:'Winter', 1:'Winter', 2:'Winter', 3:'Spring', 4:'Spring', 5:'Spring', 6:'Summer', 7:'Summer', 8:'Summer', 9:'Monsoon', 10:'Monsoon', 11:'Monsoon'})
        fig = px.box(df, x='Season', y='AQI', color='Season', title="Seasonal Air Quality Levels", category_orders={"Season": ["Winter", "Spring", "Summer", "Monsoon"]})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Date column missing (Check CSV).")

# --- TAB 4: MODEL ---
with tab4:
    st.header("Model Training")
    st.markdown("Training Random Forest on **Scaled Data**.")
    if st.button("🚀 Train Model Now", type="primary"):
        with st.spinner("Training..."):
            X = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
            y = df['AQI']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            joblib.dump(model, "aqi_model.pkl")
            st.success(f"Model Trained! R² Score: {r2_score(y_test, model.predict(X_test)):.4f}")
            st.balloons()

# --- TAB 5: PREDICT (SCALED) ---
with tab5:
    st.header("Predict Air Quality")
    try:
        model = joblib.load("aqi_model.pkl")
        X_feats = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
        
        st.info("ℹ️ **Input: Standardized Z-Scores (-5 to +5)**")
        inputs = []
        cols = st.columns(3)
        for i, f in enumerate(X_feats.columns[:9]): # Show top 9 features
            with cols[i % 3]:
                inputs.append(st.slider(f"{f} (Z-Score)", -5.0, 5.0, 0.0, 0.1))
        
        if st.button("Predict AQI", type="primary", use_container_width=True):
            full_in = inputs + [0.0] * (len(X_feats.columns) - len(inputs))
            pred = model.predict([full_in])[0]
            
            c1, c2 = st.columns([1,2])
            with c1: st.markdown(f"<h1 style='color:#10b981;font-size:60px'>{pred:.0f}</h1>", unsafe_allow_html=True)
            with c2: 
                fig = go.Figure(go.Indicator(mode="gauge+number", value=pred, gauge={'axis': {'range': [0, 500]}, 'bar': {'color': "white"}, 'steps': [{'range': [0,100], 'color': "#00b894"}, {'range': [100,200], 'color': "#fdcb6e"}, {'range': [200,500], 'color': "#d63031"}]}))
                fig.update_layout(height=250, margin=dict(t=30,b=20,l=20,r=20))
                st.plotly_chart(fig, use_container_width=True)
    except: st.warning("Train the model first.")

# --- TAB 6: MAP ---
with tab6:
    st.header("Pollution Hotspots")
    if 'City' in df.columns:
        coords = {'Delhi':(28.61,77.20), 'Mumbai':(19.07,72.87), 'Bengaluru':(12.97,77.59), 'Kolkata':(22.57, 88.36), 'Chennai':(13.08, 80.27), 'Hyderabad':(17.38, 78.48), 'Ahmedabad':(23.02, 72.57), 'Lucknow':(26.84, 80.94), 'Patna':(25.59, 85.13), 'Gurugram':(28.45, 77.02), 'Amritsar':(31.63, 74.87), 'Jaipur':(26.91, 75.78), 'Visakhapatnam':(17.68, 83.21), 'Thiruvananthapuram':(8.52, 76.93), 'Nagpur':(21.14, 79.08), 'Chandigarh':(30.73, 76.77), 'Bhopal':(23.25, 77.41), 'Shillong':(25.57, 91.89)}
        city_stats = df.groupby('City')['AQI'].mean().reset_index()
        city_stats['lat'] = city_stats['City'].map(lambda x: coords.get(x, (None,None))[0])
        city_stats['lon'] = city_stats['City'].map(lambda x: coords.get(x, (None,None))[1])
        fig = px.scatter_mapbox(city_stats.dropna(subset=['lat']), lat="lat", lon="lon", size="AQI", color="AQI", hover_name="City", zoom=3.5, color_continuous_scale="RdYlGn_r")
        fig.update_layout(mapbox_style="carto-positron", height=600)
        st.plotly_chart(fig, use_container_width=True)
    else: st.warning("City column missing (Check CSV).")

# --- TAB 7: ABOUT ---
with tab7:
    st.header("About")
    st.markdown("**WRT1** | **CMP7005 Assessment** | Student: ST20316895")
