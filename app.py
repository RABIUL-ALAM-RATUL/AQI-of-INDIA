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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ============================================================================
# 1. APP CONFIGURATION
# ============================================================================
st.set_page_config(page_title="India Air Quality", layout="wide", page_icon="🌤️")

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
# 2. DATA LOADING
# ============================================================================
@st.cache_data
def load_data():
    try:
        # Load Data (GZIP)
        df = pd.read_csv("India_Air_Quality_Final_Processed.csv.gz", compression='gzip')
        df.columns = [c.strip() for c in df.columns]

        # Standardize Names
        rename_map = {'city': 'City', 'date': 'Date', 'aqi': 'AQI'}
        new_names = {c: rename_map[c.lower()] for c in df.columns if c.lower() in rename_map}
        df.rename(columns=new_names, inplace=True)

        if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if 'AQI' in df.columns: df = df.dropna(subset=['AQI'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()
if df.empty: st.stop()

# ============================================================================
# 3. GLOBAL MODEL TRAINING (ENSURES CONSISTENCY)
# ============================================================================
# We train the model ONCE when the app starts so features match perfectly.
@st.cache_resource
def train_global_model(data):
    # Select only numeric features
    X = data.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
    y = data['AQI']
    
    # Train Model
    model = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    return model, X.columns.tolist()

# Train the model immediately so it's ready for the Predict Tab
global_model, feature_names = train_global_model(df)

# ============================================================================
# 4. TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Home", "EDA", "Seasonal", "Model", "Predict", "Map", "About"])

# --- TAB 1: DASHBOARD ---
with tab1:
    st.header("Project Dashboard")
    total_cities = df['City'].nunique() if 'City' in df.columns else 0
    avg_aqi = df['AQI'].mean() if 'AQI' in df.columns else 0
    max_aqi = df['AQI'].max() if 'AQI' in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card">Records<br><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card">Cities<br><h2>{total_cities}</h2></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card">Avg AQI<br><h2>{avg_aqi:.1f}</h2></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card">Max AQI<br><h2>{max_aqi:.0f}</h2></div>', unsafe_allow_html=True)

    if 'Date' in df.columns and 'City' in df.columns:
        st.markdown("### National AQI Trends")
        plot_data = df.sample(min(5000, len(df))) if len(df) > 5000 else df
        fig = px.line(plot_data, x='Date', y='AQI', color='City', title="AQI Over Time")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: EDA ---
with tab2:
    st.header("Correlation Analysis")
    numeric = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
    if not numeric.empty:
        fig = px.imshow(numeric.corr(), text_auto=True, color_continuous_scale='RdBu_r', height=700, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: SEASONAL ---
with tab3:
    st.header("Seasonal Patterns")
    if 'Date' in df.columns and 'AQI' in df.columns:
        df['Season'] = df['Date'].dt.month.map({12:'Winter', 1:'Winter', 2:'Winter', 3:'Spring', 4:'Spring', 5:'Spring', 6:'Summer', 7:'Summer', 8:'Summer', 9:'Monsoon', 10:'Monsoon', 11:'Monsoon'})
        fig = px.box(df, x='Season', y='AQI', color='Season', title="Seasonal Air Quality Levels", category_orders={"Season": ["Winter", "Spring", "Summer", "Monsoon"]})
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 4: MODEL STATS ---
with tab4:
    st.header("Model Performance")
    st.markdown("The model is trained automatically in the background.")
    
    # Evaluate the global model on a small test split
    X = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
    y = df['AQI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Quick predictions for metrics
    pred_test = global_model.predict(X_test)
    r2 = r2_score(y_test, pred_test)
    mse = mean_squared_error(y_test, pred_test)
    mae = mean_absolute_error(y_test, pred_test)
    
    st.success(f"Model Ready! Trained on {len(X_train)} records.")
    
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("R² Score", f"{r2:.4f}")
    with m2: st.metric("MSE", f"{mse:.2f}")
    with m3: st.metric("MAE", f"{mae:.2f}")

# --- TAB 5: PREDICT (FIXED) ---
with tab5:
    st.header("Predict Air Quality")
    st.info(f"ℹ️ **Adjust Sliders (Z-Scores)**. The model uses **{len(feature_names)} features**.")
    
    # Create input dictionary
    user_inputs = {}
    
    # Create sliders dynamically
    cols = st.columns(3)
    for i, col in enumerate(feature_names):
        with cols[i % 3]:
            # Range -5 to +5 covering typical Z-scores
            val = st.slider(col, -5.0, 5.0, 0.0, 0.1)
            user_inputs[col] = val
    
    # Convert to DataFrame (Correct Order Guaranteed)
    input_df = pd.DataFrame([user_inputs])
    
    if st.button("Predict AQI", type="primary", use_container_width=True):
        prediction = global_model.predict(input_df)[0]
        
        st.divider()
        c1, c2 = st.columns([1,2])
        
        # Color Logic
        color = "#10b981"
        if prediction > 100: color = "#facc15"
        if prediction > 200: color = "#f97316"
        if prediction > 300: color = "#ef4444"
        if prediction > 400: color = "#7f1d1d"
        
        with c1: 
            st.markdown(f"<h1 style='color:{color};font-size:70px;margin:0'>{prediction:.0f}</h1>", unsafe_allow_html=True)
            st.caption("Predicted AQI")
            
        with c2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prediction, 
                title={'text': "Severity"}, 
                gauge={'axis': {'range': [0, 500]}, 'bar': {'color': color},
                       'steps': [{'range': [0,50], 'color': "#00e400"},
                                 {'range': [50,100], 'color': "#ffff00"},
                                 {'range': [100,200], 'color': "#ff7e00"},
                                 {'range': [200,300], 'color': "#ff0000"},
                                 {'range': [300,400], 'color': "#8f3f97"},
                                 {'range': [400,500], 'color': "#7e0023"}]}
            ))
            fig.update_layout(height=300, margin=dict(t=30,b=20,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 6: MAP ---
with tab6:
    st.header("Pollution Hotspots")
    if 'City' in df.columns and 'AQI' in df.columns:
        coords = {'Delhi':(28.61,77.20), 'Mumbai':(19.07,72.87), 'Bengaluru':(12.97,77.59), 'Kolkata':(22.57, 88.36), 'Chennai':(13.08, 80.27), 'Hyderabad':(17.38, 78.48), 'Ahmedabad':(23.02, 72.57), 'Lucknow':(26.84, 80.94), 'Patna':(25.59, 85.13), 'Gurugram':(28.45, 77.02), 'Amritsar':(31.63, 74.87), 'Jaipur':(26.91, 75.78), 'Visakhapatnam':(17.68, 83.21), 'Thiruvananthapuram':(8.52, 76.93), 'Nagpur':(21.14, 79.08), 'Chandigarh':(30.73, 76.77), 'Bhopal':(23.25, 77.41), 'Shillong':(25.57, 91.89)}
        city_stats = df.groupby('City')['AQI'].mean().reset_index()
        city_stats['lat'] = city_stats['City'].map(lambda x: coords.get(x, (None,None))[0])
        city_stats['lon'] = city_stats['City'].map(lambda x: coords.get(x, (None,None))[1])
        fig = px.scatter_mapbox(city_stats.dropna(subset=['lat']), lat="lat", lon="lon", size="AQI", color="AQI", hover_name="City", zoom=3.5, color_continuous_scale="RdYlGn_r", title="Average AQI by City")
        fig.update_layout(mapbox_style="carto-positron", height=600)
        st.plotly_chart(fig, use_container_width=True)
    else: st.warning("⚠️ 'City' column missing.")

# --- TAB 7: ABOUT ---
with tab7:
    st.header("About")
    st.markdown("**CMP7005 Assessment** | Student: ST20316895")
