import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# Professional Cardiff Met Header
st.set_page_config(page_title="India Air Quality", layout="wide")
st.markdown("""
<style>
    .header {background: linear-gradient(90deg, #001f3f, #003366); padding: 35px; border-radius: 18px; 
             color: white; text-align: center; box-shadow: 0 12px 35px rgba(0,0,0,0.4); margin-bottom: 40px;}
    .header img {height: 110px; margin-right: 25px;}
    .title {font-size: 56px; font-weight: bold; margin: 0;}
    .subtitle {font-size: 28px; margin: 12px 0; color: #e2e8f0;}
    .stTabs [data-testid="stTab"] {background: #001f3f; color: white; border-radius: 12px 12px 0 0; padding: 16px 34px; font-weight: bold;}
    .stTabs [aria-selected="true"] {background: #0074D9;}
    .metric-card {background: linear-gradient(135deg, #0074D9, #001f3f); padding: 35px;
                  border-radius: 22px; color: white; text-align: center; box-shadow: 0 12px 30px rgba(0,0,0,0.3);}
</style>

<div class="header">
    <img src="https://www.cardiffmet.ac.uk/PublishingImages/logo.png" alt="Cardiff Met">
    <div class="title">India Air Quality Analysis Dashboard</div>
    <div class="subtitle">CMP7005 • ST20316895 • 2025-26</div>
</div>
""", unsafe_allow_html=True)

# Load & Merge Data from City CSVs (26 Cities)
@st.cache_data
def load_data():
    city_files = [f for f in os.listdir('.') if f.endswith('_data.csv')]
    df = pd.concat([pd.read_csv(f) for f in city_files], ignore_index=True)
    
    # Standardize columns
    df.columns = [c.strip().lower().replace(' ', '_').replace('.', '') for c in df.columns]
    
    # Fix date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Fix numeric
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
    
    fig = px.line(df.sample(min(5000, len(df))), x='date', y='aqi', color='city', title="AQI Trends")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Exploratory Data Analysis")
    numeric = df.select_dtypes(include='number').columns.drop('aqi', errors='ignore')
    fig = px.imshow(
            corr, 
            text_auto=True, 
            color_continuous_scale='RdBu_r', 
            title="Pollutant Correlation Matrix",
            height=800,  # Make it tall enough
            aspect="auto" # Adapt to container width
        )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Seasonal Patterns")
    df['Season'] = df['date'].dt.month.map({12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
                                            6:'Summer',7:'Summer',8:'Summer',9:'Monsoon',10:'Monsoon',11:'Monsoon'})
    fig = px.box(df, x='Season', y='aqi', color='Season', title="AQI by Season")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Model Training")
    X = df.select_dtypes(include='number').drop(columns=['aqi'], errors='ignore')
    y = df['aqi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if st.button("Train Random Forest Model", type="primary"):
        with st.spinner("Training 100 trees..."):
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
        # Use actual features from data
        features = X.columns.tolist()
        inputs = []
        for f in features[:6]:  # Show first 6 features
            inputs.append(st.slider(f, 0.0, 500.0, 100.0))
        
        if st.button("Predict AQI", type="primary"):
            input_arr = np.array([inputs + [50.0]*(len(features)-len(inputs))])
            pred = model.predict(input_arr)[0]
            st.markdown(f"<h1 style='color:#10b981'>Predicted AQI: {pred:.1f}</h1>", unsafe_allow_html=True)
    except:
        st.info("Train the model first in 'Model' tab")

with tab6:
    st.header("Pollution Hotspots")
    city_coords = {'Delhi':(28.61,77.20), 'Mumbai':(19.07,72.87), 'Bengaluru':(12.97,77.59)}
    city_aqi = df.groupby('city')['aqi'].mean().round(0).reset_index()
    city_aqi['lat'] = city_aqi['city'].map({k:v[0] for k,v in city_coords.items()})
    city_aqi['lon'] = city_aqi['city'].map({k:v[1] for k,v in city_coords.items()})
    
    fig = px.scatter_mapbox(city_aqi.dropna(), lat="lat", lon="lon", size="aqi", color="aqi",
                            hover_name="city", zoom=4, title="AQI Hotspots")
    fig.update_layout(mapbox_style="carto-positron", height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab7:
    st.header("About This Project")
    st.markdown("""
    **CMP7005 – Programming for Data Analysis**  \nStudent ID: ST20316895  \n2025-26  \nCardiff Metropolitan University
    """)
    st.image("https://www.cardiffmet.ac.uk/PublishingImages/logo.png", width=300)
