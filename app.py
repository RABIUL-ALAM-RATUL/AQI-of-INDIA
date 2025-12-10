import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for Professional Beauty
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
    }
    h1, h2, h3 {
        color: #f8fafc !important;
        font-family: 'Arial Black', sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .stButton > button {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        border-radius: 10px;
        font-weight: bold;
        border: none;
        padding: 8px 16px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #059669, #047857);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# App Title
st.title("🌫️ India Air Quality Analytics Dashboard")
st.markdown("**Professional Analysis & ML Forecasting | CMP7005 Project**")

# Sidebar Navigation (9 Pages)
st.sidebar.title("Navigation")
pages = [
    "📊 Home & Overview",
    "📁 Data Upload & Preprocess",
    "🔍 Exploratory Data Analysis",
    "⚙️ Feature Engineering",
    "🤖 Model Training",
    "🔮 AQI Prediction (ML)",
    "🗺️ Geospatial Insights",
    "📈 Advanced Analytics",
    "ℹ️ Project Info"
]
page = st.sidebar.selectbox("Select Page", pages)

# Sample Data for Demo (Replace with real data)
@st.cache_data
def load_demo_data():
    np.random.seed(42)
    n = 5000
    return pd.DataFrame({
        'date': pd.date_range('2015-01-01', periods=n, freq='D'),
        'city': np.random.choice(['Delhi', 'Mumbai', 'Bengaluru', 'Chennai', 'Kolkata'], n),
        'pm25': np.random.normal(120, 60, n).clip(0, 400),
        'pm10': np.random.normal(200, 80, n).clip(0, 500),
        'no2': np.random.normal(50, 20, n).clip(0, 150),
        'so2': np.random.normal(25, 10, n).clip(0, 80),
        'co': np.random.normal(1.5, 0.5, n).clip(0, 5),
        'o3': np.random.normal(60, 25, n).clip(0, 150),
        'aqi': np.random.normal(180, 80, n).clip(0, 500),
        'aqi_bucket': np.random.choice(['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe'], n)
    })

df = load_demo_data()

# Page 1: Home & Overview
if page == "📊 Home & Overview":
    st.header("Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">Total Records<br><h1>{:,}</h1></div>'.format(len(df)), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">Cities Covered<br><h1>{}</h1></div>'.format(df['city'].nunique()), unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">Avg AQI<br><h1>{:.1f}</h1></div>'.format(df['aqi'].mean()), unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">Highest AQI<br><h1>{:.0f}</h1></div>'.format(df['aqi'].max()), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(df, x='date', y='aqi', color='city', title="AQI Trends Over Time")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, x='aqi_bucket', y='pm25', title="PM2.5 by AQI Category")
        st.plotly_chart(fig, use_container_width=True)

# Page 2: Data Upload & Preprocess
elif page == "📁 Data Upload & Preprocess":
    st.header("Data Upload & Preprocessing")
    
    uploaded_file = st.file_uploader("Upload Air Quality CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        
        if st.button("Preprocess Data"):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            numeric_cols = df.select_dtypes(include='number').columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='time').ffill().bfill()
            st.success("Preprocessing Complete!")
            st.dataframe(df.head())
            joblib.dump(df, 'processed_data.pkl')

# Page 3: Exploratory Data Analysis
elif page == "🔍 Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.imshow(df.corr(numeric_only=True), title="Pollutant Correlations")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(df, x='pm25', y='aqi', color='aqi_bucket', title="PM2.5 vs AQI")
        st.plotly_chart(fig, use_container_width=True)

# Page 4: Feature Engineering
elif page == "⚙️ Feature Engineering":
    st.header("Feature Engineering")
    
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['season'] = df['month'].map({12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Monsoon',10:'Monsoon',11:'Monsoon'})
    df['is_weekend'] = df['date'].dt.weekday >= 5
    
    st.dataframe(df[['date', 'aqi', 'year', 'season', 'is_weekend']].head())
    
    fig = px.box(df, x='season', y='aqi', title="AQI by Season")
    st.plotly_chart(fig, use_container_width=True)

# Page 5: Model Training
elif page == "🤖 Model Training":
    st.header("Model Training")
    
    X = df.drop(['aqi', 'date', 'city', 'aqi_bucket'], axis=1, errors='ignore')
    y = df['aqi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if st.button("Train Linear Regression Baseline"):
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        st.success(f"Baseline R²: {r2:.4f}")
        joblib.dump(model, 'baseline_model.pkl')
    
    if st.button("Train Random Forest"):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        st.success(f"RF R²: {r2:.4f}")
        joblib.dump(model, 'rf_model.pkl')

# Page 6: AQI Prediction (ML)
elif page == "🔮 AQI Prediction (ML)":
    st.header("AQI Prediction Tool")
    
    model = joblib.load('rf_model.pkl')
    scaler = StandardScaler()
    X_sample = df.drop(['aqi', 'date', 'city', 'aqi_bucket'], axis=1, errors='ignore')
    X_scaled = scaler.fit_transform(X_sample)
    
    pm25 = st.slider("PM2.5 (µg/m³)", 0, 500, 100)
    pm10 = st.slider("PM10 (µg/m³)", 0, 500, 200)
    no2 = st.slider("NO2 (µg/m³)", 0, 200, 50)
    
    if st.button("Predict AQI", type="primary"):
        input_data = np.array([[pm25, pm10, no2] + [0]*(X_scaled.shape[1]-3)])
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        
        col1, col2 = st.columns(2)
        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=pred,
                gauge={'axis': {'range': [0, 500]}, 'bar': {'color': "#10b981"}},
                title={'text': "Predicted AQI"}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
        with col2:
            if pred <= 50:
                st.success("Good Air Quality")
            elif pred <= 100:
                st.info("Satisfactory")
            elif pred <= 200:
                st.warning("Moderate")
            elif pred <= 300:
                st.error("Poor")
            else:
                st.error("Severe – Stay Indoors!")

# Page 7: Geospatial Insights
elif page == "🗺️ Geospatial Insights":
    st.header("Geospatial Pollution Hotspots")
    
    # Sample coordinates for cities
    coords = {
        'Delhi': (28.6139, 77.2090), 'Mumbai': (19.0760, 72.8777),
        'Bengaluru': (12.9716, 77.5946), 'Chennai': (13.0827, 80.2707),
        'Kolkata': (22.5726, 88.3639)
    }
    df_geo = df.copy()
    df_geo['lat'] = df_geo['city'].map(lambda c: coords.get(c, [0])[0])
    df_geo['lon'] = df_geo['city'].map(lambda c: coords.get(c, [0])[1])
    
    fig = px.scatter_geo(df_geo, lon='lon', lat='lat', color='aqi', size='aqi',
                         hover_name='city', title="AQI Hotspots Map")
    st.plotly_chart(fig, use_container_width=True)

# Page 8: Advanced Analytics
elif page == "📈 Advanced Analytics":
    st.header("Advanced Insights")
    
    fig_corr = px.imshow(df.corr(numeric_only=True), title="Advanced Correlation Analysis")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    fig_trend = px.line(df, x='date', y='aqi', color='city', title="Long-Term Trends")
    st.plotly_chart(fig_trend, use_container_width=True)

# Page 9: Project Info
elif page == "ℹ️ Project Info":
    st.header("Project Information")
    st.markdown("""
    **CMP7005 Programming for Data Analysis**  
    - Student ID: ST20316895  
    - Module: Programming for Data Analysis  
    - Year: 2025-26  
    - Dataset: India Air Quality (2015-2020)  
    - Models: Linear Regression, Random Forest  
    - Features: Preprocessing, EDA, ML Prediction, Geospatial  
    - Built with Streamlit & Plotly  
    """)
    st.image("https://via.placeholder.com/800x400?text=Air+Quality+Project", use_column_width=True)

if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.markdown("**© 2025 Cardiff Metropolitan University**")
