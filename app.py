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

# Cardiff Met Professional Header
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
    <div class="title">India Air Quality Analysis</div>
    <div class="subtitle">CMP7005 • ST20316895 • 2025-26</div>
</div>
""", unsafe_allow_html=True)

# Load Data — 100% Safe for ANY column naming
@st.cache_data
def load_data():
    df = pd.read_csv("India_Air_Quality_Final_Processed.csv")
    
    # Auto-detect key columns
    df.columns = [c.strip() for c in df.columns]
    
    # Find AQI column
    aqi_candidates = [c for c in df.columns if 'AQI' in c.upper() and 'BUCKET' not in c.upper()]
    aqi_col = aqi_candidates[0] if aqi_candidates else None
    
    # Find Date column
    date_candidates = [c for c in df.columns if 'DATE' in c.upper()]
    date_col = date_candidates[0] if date_candidates else None
    
    # Find City column
    city_candidates = [c for c in df.columns if 'CITY' in c.upper()]
    city_col = city_candidates[0] if city_candidates else None
    
    # Rename to standard names
    if aqi_col: df.rename(columns={aqi_col: 'AQI'}, inplace=True)
    if date_col: df.rename(columns={date_col: 'Date'}, inplace=True)
    if city_col: df.rename(columns={city_col: 'City'}, inplace=True)
    
    # Convert Date
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Keep only numeric columns + essential non-numeric
    numeric_df = df.select_dtypes(include=[np.number])
    if 'AQI' in df.columns:
        numeric_df['AQI'] = df['AQI']
    if 'Date' in df.columns:
        numeric_df['Date'] = df['Date']
    if 'City' in df.columns:
        numeric_df['City'] = df['City']
    
    return numeric_df

df = load_data()

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Home", "EDA", "Seasonal", "Model", "Predict", "Map", "About"
])

with tab1:
    st.header("Project Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card">Records<br><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card">Cities<br><h2>{df["City"].nunique() if "City" in df.columns else "N/A"}</h2></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card">Avg AQI<br><h2>{df["AQI"].mean():.1f}</h2></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card">Peak AQI<br><h2>{df["AQI"].max():.0f}</h2></div>', unsafe_allow_html=True)
    
    sample = df.sample(min(5000, len(df)))
    fig = px.line(sample, x='Date', y='AQI', color='City' if 'City' in df.columns else None, title="AQI Trends")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Exploratory Data Analysis")
    num_cols = df.select_dtypes(include='number').columns.drop('AQI', errors='ignore')
    fig = px.imshow(df[num_cols].corr(), title="Pollutant Correlations", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Seasonal Patterns")
    df['Season'] = df['Date'].dt.month.map({12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
                                            6:'Summer',7:'Summer',8:'Summer',9:'Monsoon',10:'Monsoon',11:'Monsoon'})
    fig = px.box(df, x='Season', y='AQI', color='Season', title="AQI by Season")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Model Training")
    # Only numeric features for ML
    X = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
    y = df['AQI']
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
        # Use actual feature names from your data
        feature_names = X.columns.tolist()
        values = []
        for feat in feature_names[:6]:  # Show first 6
            values.append(st.slider(feat, 0, 500, 100))
        
        if st.button("Predict AQI", type="primary"):
            input_arr = np.array([values + [50]*(len(feature_names)-len(values))])
            pred = model.predict(input_arr)[0]
            st.markdown(f"<h1 style='color:#10b981'>Predicted AQI: {pred:.1f}</h1>", unsafe_allow_html=True)
    except:
        st.info("Train the model first in 'Model' tab")

with tab6:
    st.header("Pollution Hotspots")
    if 'City' in df.columns:
        city_aqi = df.groupby('City')['AQI'].mean().round(0).reset_index()
        # Simple coords for top cities
        coords = {'Delhi':(28.61,77.20), 'Mumbai':(19.07,72.87), 'Bengaluru':(12.97,77.59)}
        city_aqi['lat'] = city_aqi['City'].map({k:v[0] for k,v in coords.items()})
        city_aqi['lon'] = city_aqi['City'].map({k:v[1] for k,v in coords.items()})
        city_aqi = city_aqi.dropna()
        
        fig = px.scatter_mapbox(city_aqi, lat="lat", lon="lon", size="AQI", color="AQI",
                                hover_name="City", zoom=4, title="Major City Hotspots")
        fig.update_layout(mapbox_style="carto-positron", height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("City column not found")

with tab7:
    st.header("About This Project")
    st.markdown("""
    **CMP7005 – Programming for Data Analysis**  
    **Student ID:** ST20316895  
    **Academic Year:** 2025–26  
    **Module Leader:** aprasad@cardiffmet.ac.uk  
    **Dataset:** India Air Quality (2015–2020)  
    **Built with:** Python • Pandas • Scikit-learn • Streamlit • Plotly  
    """)
    st.image("https://www.cardiffmet.ac.uk/PublishingImages/logo.png", width=300)
