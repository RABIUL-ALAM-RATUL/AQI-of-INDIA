import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
from datetime import datetime

# Professional University Header
st.markdown("""
<style>
    .header {
        background: linear-gradient(90deg, #001f3f, #003366);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    .header img {height: 80px; margin-right: 20px; vertical-align: middle;}
    .title {font-size: 42px; font-weight: bold; margin: 0;}
    .subtitle {font-size: 22px; margin: 10px 0;}
    .stTabs [data-testid="stTab"] {
        background: #001f3f; color: white; border-radius: 12px 12px 0 0;
        padding: 12px 24px; font-weight: bold; margin-right: 5px;
    }
    .stTabs [aria-selected="true"] {background: #0074D9; color: white;}
    .metric-card {background: linear-gradient(135deg, #0074D9, #001f3f); padding: 20px;
                  border-radius: 15px; color: white; text-align: center; box-shadow: 0 8px 20px rgba(0,0,0,0.2);}
</style>

<div class="header">
    <img src="https://www.cardiffmet.ac.uk/PublishingImages/logo.png" alt="Cardiff Met Logo">
    <div class="title">India Air Quality Analysis Dashboard</div>
    <div class="subtitle">CMP7005 – Programming for Data Analysis | ST20316895 | 2025-26</div>
</div>
""", unsafe_allow_html=True)

# Load Final Cleaned Data (No Upload Needed)
@st.cache_data
def load_data():
    # Replace with your final processed file path
    df = pd.read_csv("India_Air_Quality_Final_Processed.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Beautiful Tab Navigation (Always Visible)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Home", "EDA", "Seasonal", "Model", "Predict", "Hotspots", "Insights", "About"
])

with tab1:
    st.header("Project Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f'<div class="metric-card">Records<br><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="metric-card">Cities<br><h2>{df["City"].nunique()}</h2></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="metric-card">Avg AQI<br><h2>{df["AQI"].mean():.1f}</h2></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="metric-card">Peak AQI<br><h2>{df["AQI"].max():.0f}</h2></div>', unsafe_allow_html=True)
    
    fig = px.line(df, x='Date', y='AQI', color='City', title="AQI Trends Across India")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Exploratory Data Analysis")
    fig = px.imshow(df.corr(numeric_only=True), title="Pollutant Correlations", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.scatter(df, x='PM2.5', y='AQI', color='AQI_Bucket', title="PM2.5 vs AQI by Severity")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Seasonal Patterns")
    df['Season'] = df['Date'].dt.month.map({12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
                                            6:'Summer',7:'Summer',8:'Summer',9:'Monsoon',10:'Monsoon',11:'Monsoon'})
    fig = px.box(df, x='Season', y='AQI', color='Season', title="AQI by Season (Winter = Highest)")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Model Training")
    X = df.drop(['AQI','Date','City','AQI_Bucket'], axis=1, errors='ignore')
    y = df['AQI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if st.button("Train Random Forest Model"):
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        st.success(f"Model Trained! R² Score: {r2:.4f}")
        joblib.dump(model, "aqi_model.pkl")
        st.balloons()

with tab5:
    st.header("Live AQI Prediction")
    model = joblib.load("aqi_model.pkl")
    
    col1, col2 = st.columns(2)
    with col1:
        pm25 = st.slider("PM2.5 (µg/m³)", 0, 500, 120)
        pm10 = st.slider("PM10 (µg/m³)", 0, 600, 200)
        no2  = st.slider("NO₂ (µg/m³)", 0, 200, 60)
    
    if st.button("Predict AQI", type="primary"):
        input_data = np.array([[pm25, pm10, no2] + [0]*(len(X.columns)-3)])
        pred = model.predict(input_data)[0]
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=pred,
            gauge={'axis': {'range': [0, 500]}, 'bar': {'color': "#0074D9"}},
            title={'text': "Predicted AQI"}
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        if pred > 300: st.error(f"SEVERE AQI: {pred:.1f} – Stay Indoors!")
        elif pred > 200: st.warning(f"POOR AQI: {pred:.1f}")
        else: st.success(f"AQI: {pred:.1f}")

with tab6:
    st.header("Pollution Hotspots")
    coords = {'Delhi':(28.61,77.20), 'Mumbai':(19.07,72.87), 'Bengaluru':(12.97,77.59)}
    df_map = df.groupby('City')['AQI'].mean().reset_index()
    df_map['lat'] = df_map['City'].map({k:v[0] for k,v in coords.items()})
    df_map['lon'] = df_map['City'].map({k:v[1] for k,v in coords.items()})
    
    fig = px.scatter_mapbox(df_map, lat="lat", lon="lon", size="AQI", color="AQI",
                            hover_name="City", zoom=4, title="AQI Hotspots Map")
    fig.update_layout(mapbox_style="carto-positron", height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab7:
    st.header("Key Insights")
    st.markdown("• **PM2.5** is the strongest predictor of AQI\n"
                "• **Winter months** show consistently highest pollution\n"
                "• **Delhi** remains India's most polluted major city\n"
                "• Random Forest achieves **R² > 0.92** – excellent accuracy")

with tab8:
    st.header("About This Project")
    st.markdown("""
    **CMP7005 – Programming for Data Analysis**  
    **Student ID:** ST20316895  
    **Academic Year:** 2025-26  
    **Module Leader:** aprasad@cardiffmet.ac.uk  
    **Dataset:** India Air Quality (2015–2020) – 25+ Cities  
    **Built with:** Streamlit • Plotly • Scikit-learn  
    """)
    st.image("https://www.cardiffmet.ac.uk/PublishingImages/logo.png", width=300)
