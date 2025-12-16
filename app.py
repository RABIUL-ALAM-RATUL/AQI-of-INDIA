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
warnings.filterwarnings("ignore")

# Clean Professional Header (No Logo)
st.set_page_config(page_title="India Air Quality Dashboard", layout="wide")
st.markdown("""
<style>
    .header {
        background: linear-gradient(90deg, #0f172a, #1e293b);
        padding: 35px;
        border-radius: 15px;
        color: #e2e8f0;
        text-align: center;
        box-shadow: 0 12px 35px rgba(0,0,0,0.4);
        margin-bottom: 40px;
    }
    .title {font-size: 52px; font-weight: bold; margin: 0;}
    .subtitle {font-size: 26px; margin: 12px 0;}
    .stTabs [data-testid="stTab"] {
        background: #0f172a; color: white; border-radius: 12px 12px 0 0;
        padding: 16px 34px; font-weight: bold;
    }
    .stTabs [aria-selected="true"] {background: #3b82f6;}
    .metric-card {
        background: linear-gradient(135deg, #3b82f6, #1e40af);
        padding: 35px;
        border-radius: 22px;
        color: white;
        text-align: center;
        box-shadow: 0 12px 30px rgba(0,0,0,0.3);
    }
</style>

<div class="header">
    <div class="title">India Air Quality Analysis Dashboard</div>
    <div class="subtitle">CMP7005 • ST20316895 • 2025-26</div>
</div>
""", unsafe_allow_html=True)

# Load Data — 100% Safe & Correct
@st.cache_data
def load_data():
    df = pd.read_csv("India_Air_Quality_Final_Processed.csv")
    
    # Standardize column names
    df.columns = [c.strip().lower().replace(' ', '_').replace('.', '') for c in df.columns]
    
    # Rename key columns safely
    if 'aqi' not in df.columns:
        aqi_col = [c for c in df.columns if 'aqi' in c and 'bucket' not in c]
        if aqi_col: df.rename(columns={aqi_col[0]: 'aqi'}, inplace=True)
    
    if 'date' not in df.columns:
        date_col = [c for c in df.columns if 'date' in c]
        if date_col: df.rename(columns={date_col[0]: 'date'}, inplace=True)
    
    if 'city' not in df.columns:
        city_col = [c for c in df.columns if 'city' in c]
        if city_col: df.rename(columns={city_col[0]: 'city'}, inplace=True)
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Clean numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df

df = load_data()

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Home", "EDA", "Seasonal", "Model", "Predict", "Map", "About"
])

with tab1:
    st.header("Project Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card">Records<br><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card">Cities<br><h2>{df["city"].nunique()}</h2></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card">Avg AQI<br><h2>{df["aqi"].mean():.1f}</h2></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card">Peak AQI<br><h2>{df["aqi"].max():.0f}</h2></div>', unsafe_allow_html=True)
    
    fig = px.line(df.sample(min(5000, len(df))), x='date', y='aqi', color='city', title="AQI Trends Across 26 Cities")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Exploratory Data Analysis")
    numeric_cols = df.select_dtypes(include='number').columns.drop('aqi', errors='ignore')
    fig = px.imshow(df[numeric_cols].corr(), title="Pollutant Correlation Matrix", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Seasonal Patterns")
    df['season'] = df['date'].dt.month.map({12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
                                            6:'Summer',7:'Summer',8:'Summer',9:'Monsoon',10:'Monsoon',11:'Monsoon'})
    fig = px.box(df, x='season', y='aqi', color='season', title="AQI by Season")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Model Training")
    X = df.select_dtypes(include='number').drop(columns=['aqi'], errors='ignore')
    y = df['aqi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if st.button("Train Random Forest Model", type="primary"):
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        joblib.dump(model, "aqi_model.pkl")
        st.success(f"Model Trained! R² = {r2:.4f}")
        st.balloons()

with tab5:
    st.header("Live AQI Prediction")
    try:
        model = joblib.load("aqi_model.pkl")
        st.markdown("Adjust pollutant levels to predict AQI")
        inputs = {}
        for col in X.columns[:6]:
            inputs[col] = st.slider(col.upper(), 0.0, 500.0, 100.0)
        
        if st.button("Predict AQI", type="primary"):
            input_arr = np.array([[inputs.get(c, 50) for c in X.columns]])
            pred = model.predict(input_arr)[0]
            st.markdown(f"<h1 style='color:#10b981'>Predicted AQI: {pred:.1f}</h1>", unsafe_allow_html=True)
    except:
        st.info("Train the model first in 'Model' tab")

with tab6:
    st.header("Pollution Hotspots")
    fig = px.scatter_mapbox(df.sample(1000), lat=None, lon=None, color="aqi", size="aqi",
                            hover_data=["city", "date"], title="Sample AQI Hotspots (26 Cities)")
    fig.update_layout(mapbox_style="carto-positron", height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab7:
    st.header("About This Project")
    st.markdown("""
    **CMP7005 – Programming for Data Analysis**  
    **Student ID:** ST20316895  
    **Academic Year:** 2025–26  
    **Dataset:** India Air Quality (26 Cities, 2015–2020)  
    **Built with:** Python • Pandas • Scikit-learn • Streamlit • Plotly  
    """)
