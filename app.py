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
warnings.filterwarnings("ignore")

# Beautiful Cardiff Met Header
st.set_page_config(page_title="India Air Quality", layout="wide")
st.markdown("""
<style>
    .header {background: linear-gradient(90deg, #001f3f, #003366); padding: 30px; border-radius: 15px; 
             color: white; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.4); margin-bottom: 40px;}
    .header img {height: 100px; margin-right: 20px;}
    .title {font-size: 52px; font-weight: bold; margin: 0;}
    .subtitle {font-size: 26px; margin: 10px 0;}
    .stTabs [data-testid="stTab"] {background: #001f3f; color: white; border-radius: 12px 12px 0 0; padding: 16px 32px; font-weight: bold;}
    .stTabs [aria-selected="true"] {background: #0074D9;}
    .metric-card {background: linear-gradient(135deg, #0074D9, #001f3f); padding: 30px;
                  border-radius: 20px; color: white; text-align: center; box-shadow: 0 10px 25px rgba(0,0,0,0.3);}
</style>

<div class="header">
    <img src="https://www.cardiffmet.ac.uk/PublishingImages/logo.png" alt="Cardiff Met">
    <div class="title">India Air Quality Analysis</div>
    <div class="subtitle">CMP7005 • ST20316895 • 2025-26</div>
</div>
""", unsafe_allow_html=True)

# Load Data Safely (Handles any column name)
@st.cache_data
def load_data():
    df = pd.read_csv("India_Air_Quality_Final_Processed.csv")
    
    # Fix date column (find any column with 'date' in name)
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if date_cols:
        df['Date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
        df.drop(columns=date_cols, inplace=True)
    
    # Fix city column
    city_cols = [col for col in df.columns if 'city' in col.lower()]
    if city_cols:
        df['City'] = df[city_cols[0]]
        df.drop(columns=city_cols, inplace=True)
    
    return df

df = load_data()

# Tab Navigation
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Home", "EDA", "Seasonal", "Model", "Predict", "Map", "Insights", "About"
])

with tab1:
    st.header("Project Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card">Records<br><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card">Cities<br><h2>{df["City"].nunique()}</h2></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card">Avg AQI<br><h2>{df["AQI"].mean():.1f}</h2></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card">Peak AQI<br><h2>{df["AQI"].max():.0f}</h2></div>', unsafe_allow_html=True)
    
    fig = px.line(df.sample(min(5000, len(df))), x='Date', y='AQI', color='City', title="AQI Trends")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Exploratory Data Analysis")
    numeric_cols = df.select_dtypes(include='number').columns
    fig = px.imshow(df[numeric_cols].corr(), title="Pollutant Correlation Matrix", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Seasonal Patterns")
    df['Season'] = df['Date'].dt.month.map({12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
                                            6:'Summer',7:'Summer',8:'Summer',9:'Monsoon',10:'Monsoon',11:'Monsoon'})
    fig = px.box(df, x='Season', y='AQI', color='Season', title="AQI by Season")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Model Training")
    X = df.drop(['AQI','Date','City','AQI_Bucket'], axis=1, errors='ignore')
    y = df['AQI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if st.button("Train Random Forest"):
        with st.spinner("Training..."):
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            r2 = r2_score(y_test, pred)
            joblib.dump(model, "aqi_model.pkl")
        st.success(f"R² = {r2:.4f}")
        st.balloons()

with tab5:
    st.header("Live Prediction")
    try:
        model = joblib.load("aqi_model.pkl")
        pm25 = st.slider("PM2.5", 0, 500, 120)
        pm10 = st.slider("PM10", 0, 600, 220)
        no2  = st.slider("NO₂", 0, 200, 65)
        
        if st.button("Predict", type="primary"):
            input_data = np.array([[pm25, pm10, no2] + [50]*(len(X.columns)-3)])
            pred = model.predict(input_data)[0]
            st.markdown(f"<h1 style='color:#10b981'>Predicted AQI: {pred:.1f}</h1>", unsafe_allow_html=True)
    except:
        st.info("Train model first")

with tab6:
    st.header("Hotspots")
    city_coords = {'Delhi':(28.61,77.20), 'Mumbai':(19.07,72.87), 'Bengaluru':(12.97,77.59)}
    city_aqi = df.groupby('City')['AQI'].mean().round(0).reset_index()
    city_aqi['lat'] = city_aqi['City'].map({k:v[0] for k,v in city_coords.items()})
    city_aqi['lon'] = city_aqi['City'].map({k:v[1] for k,v in city_coords.items()})
    
    fig = px.scatter_mapbox(city_aqi.dropna(), lat="lat", lon="lon", size="AQI", color="AQI",
                            hover_name="City", zoom=4, title="AQI Hotspots")
    fig.update_layout(mapbox_style="carto-positron", height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab7:
    st.header("Key Insights")
    st.markdown("""
    - PM2.5 dominates AQI prediction
    - Winter months show highest pollution
    - Delhi remains worst major city
    - Random Forest achieves excellent accuracy
    """)

with tab8:
    st.header("About")
    st.markdown("**CMP7005 – Programming for Data Analysis**  \nStudent ID: ST20316895  \n2025-26")
    st.image("https://www.cardiffmet.ac.uk/PublishingImages/logo.png", width=300)
