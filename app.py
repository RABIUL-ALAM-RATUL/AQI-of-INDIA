# ============================================================================
# INDIA AIR QUALITY ANALYSIS APP (CMP7005 ASSESSMENT)
# DEVELOPED BY: MD RABIUL ALAM
# STUDENT ID: ST20316895
# ============================================================================

import streamlit as st                  # Framework for building the web app
import pandas as pd                     # Library for data manipulation
import numpy as np                      # Library for numerical operations
import plotly.express as px             # Library for quick, interactive charts
import plotly.graph_objects as go       # Library for detailed custom charts
from sklearn.model_selection import train_test_split # Tool to split data for ML
from sklearn.ensemble import RandomForestRegressor   # The ML algorithm we are using
from sklearn.metrics import r2_score    # Metric to evaluate model accuracy
import joblib                           # Tool to save/load trained models
import warnings                         # Tool to suppress annoying warnings

# Suppress warnings to keep the app interface clean
warnings.filterwarnings("ignore")

# ============================================================================
# 1. APP CONFIGURATION & STYLING
# ============================================================================

# Configure the page title, layout, and icon
st.set_page_config(page_title="India Air Quality", layout="wide", page_icon="🌤️")

# Apply Custom CSS for a Professional "Cardiff Met" Look
st.markdown("""
<style>
    /* Style the main header with a gradient background */
    .header {
        background: linear-gradient(90deg, #001f3f, #003366); 
        padding: 35px; 
        border-radius: 18px; 
        color: white; 
        text-align: center; 
        box-shadow: 0 12px 35px rgba(0,0,0,0.4); 
        margin-bottom: 40px;
    }
    /* Style the logo inside the header */
    .header img {height: 110px; margin-right: 25px;}
    /* Style the main title text */
    .title {font-size: 56px; font-weight: bold; margin: 0;}
    /* Style the subtitle text */
    .subtitle {font-size: 28px; margin: 12px 0; color: #e2e8f0;}
    /* Style the tabs to look like professional navigation buttons */
    .stTabs [data-testid="stTab"] {
        background: #001f3f; 
        color: white; 
        border-radius: 12px 12px 0 0; 
        padding: 16px 34px; 
        font-weight: bold;
    }
    /* Highlight the active tab */
    .stTabs [aria-selected="true"] {background: #0074D9;}
    /* Style the metric cards (Total Records, Avg AQI, etc.) */
    .metric-card {
        background: linear-gradient(135deg, #0074D9, #001f3f); 
        padding: 35px;
        border-radius: 22px; 
        color: white; 
        text-align: center; 
        box-shadow: 0 12px 30px rgba(0,0,0,0.3);
    }
</style>

<div class="header">
    <img src="https://www.cardiffmet.ac.uk/PublishingImages/logo.png" alt="Cardiff Met">
    <div class="title">India Air Quality Analysis</div>
    <div class="subtitle">CMP7005 • ST20316895 • 2025-26</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 2. DATA LOADING & CLEANING (ROBUST)
# ============================================================================

# Cache the data so it doesn't reload every time you click a button
@st.cache_data
def load_data():
    try:
        # Load the CSV file
        df = pd.read_csv("India_Air_Quality_Final_Processed.csv")
        
        # Clean up column names (remove spaces)
        df.columns = [c.strip() for c in df.columns]
        
        # Robust Column Renaming: Handle case variations (e.g., "AQI" vs "aqi")
        # Find column containing 'AQI' and rename it to standard 'AQI'
        aqi_col = next((c for c in df.columns if 'AQI' in c.upper() and 'BUCKET' not in c.upper()), None)
        if aqi_col: df.rename(columns={aqi_col: 'AQI'}, inplace=True)
        
        # Find column containing 'DATE' and rename it to 'Date'
        date_col = next((c for c in df.columns if 'DATE' in c.upper()), None)
        if date_col: df.rename(columns={date_col: 'Date'}, inplace=True)
        
        # Find column containing 'CITY' and rename it to 'City'
        city_col = next((c for c in df.columns if 'CITY' in c.upper()), None)
        if city_col: df.rename(columns={city_col: 'City'}, inplace=True)
        
        # Convert Date column to proper datetime objects
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Drop columns that are non-numeric and not essential for display
        # We keep Date, City, AQI, and all other numeric columns
        keep_cols = ['Date', 'City', 'AQI'] + df.select_dtypes(include='number').columns.tolist()
        # Filter dataframe to only kept columns (remove duplicates)
        df = df[list(set(keep_cols))].copy()
        
        # CRITICAL: Drop any rows where the Target (AQI) is missing
        df = df.dropna(subset=['AQI'])
        
        # Fill any remaining missing values in features with the median (Safe for ML)
        numeric_cols = df.select_dtypes(include='number').columns.drop('AQI', errors='ignore')
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        return df
    except Exception as e:
        # If loading fails, stop the app and show error
        st.error(f"Error loading data: {e}. Please ensure 'India_Air_Quality_Final_Processed.csv' is in the folder.")
        return pd.DataFrame()

# Load data into memory
df = load_data()

# Stop app if dataframe is empty
if df.empty:
    st.stop()

# ============================================================================
# 3. MAIN APP TABS
# ============================================================================

# Create navigation tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Home", "EDA", "Seasonal", "Model", "Predict", "Map", "About"
])

# --- TAB 1: PROJECT DASHBOARD ---
with tab1:
    st.header("Project Dashboard")
    
    # Create 4 columns for key metrics
    c1, c2, c3, c4 = st.columns(4)
    # Display HTML-styled metric cards
    with c1: st.markdown(f'<div class="metric-card">Records<br><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card">Cities<br><h2>{df["City"].nunique()}</h2></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card">Avg AQI<br><h2>{df["AQI"].mean():.1f}</h2></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card">Peak AQI<br><h2>{df["AQI"].max():.0f}</h2></div>', unsafe_allow_html=True)
    
    # Plot National Trends (Sampled for speed)
    fig = px.line(df.sample(min(5000, len(df))), x='Date', y='AQI', color='City', title="National AQI Trends")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: EXPLORATORY DATA ANALYSIS (EDA) ---
with tab2:
    st.header("Exploratory Data Analysis")
    
    # Select only numeric columns for correlation
    numeric = df.select_dtypes(include='number').columns.drop('AQI', errors='ignore')
    
    # Plot Correlation Matrix Heatmap
    fig = px.imshow(df[numeric].corr(), title="Pollutant Correlations", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: SEASONAL ANALYSIS ---
with tab3:
    st.header("Seasonal Patterns")
    
    # Feature Engineering: Map Month numbers to Season names
    df['Season'] = df['Date'].dt.month.map({
        12:'Winter', 1:'Winter', 2:'Winter',
        3:'Spring', 4:'Spring', 5:'Spring',
        6:'Summer', 7:'Summer', 8:'Summer',
        9:'Monsoon', 10:'Monsoon', 11:'Monsoon'
    })
    
    # Plot Box Plot of AQI by Season
    fig = px.box(df, x='Season', y='AQI', color='Season', title="AQI by Season")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 4: MODEL TRAINING ---
with tab4:
    st.header("Model Training")
    
    # Define Features (X) and Target (y)
    # X includes all numeric columns except AQI
    X = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
    y = df['AQI']
    
    # Button to train model
    if st.button("Train Random Forest Model", type="primary"):
        with st.spinner("Training..."):
            # Initialize Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            # Train model on FULL dataset for the app
            model.fit(X, y)
            # Calculate accuracy on training data (R2 Score)
            pred = model.predict(X)
            r2 = r2_score(y, pred)
            # Save model to file for later use
            joblib.dump(model, "aqi_model.pkl")
            
        # Success message
        st.success(f"Model Trained! R² = {r2:.4f}")
        st.balloons()

# --- TAB 5: LIVE PREDICTION ---
with tab5:
    st.header("Live AQI Prediction")
    try:
        # Load the trained model
        model = joblib.load("aqi_model.pkl")
        
        # Get feature names from the dataset used for training
        X_features = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
        features = X_features.columns.tolist()
        
        inputs = []
        # Create sliders for the top 6 features to keep UI clean
        # (You can expand this loop to show all features if needed)
        for f in features[:6]:
            val = st.slider(f, 0, 500, 100) # Default value 100
            inputs.append(val)
        
        # Button to predict
        if st.button("Predict AQI", type="primary"):
            # Prepare input array (pad with default '50' for any hidden features)
            # This ensures the model receives the exact number of inputs it expects
            full_inputs = inputs + [50]*(len(features)-len(inputs))
            input_arr = np.array([full_inputs])
            
            # Make prediction
            pred = model.predict(input_arr)[0]
            
            # Display Result
            st.markdown(f"<h1 style='color:#10b981'>Predicted AQI: {pred:.1f}</h1>", unsafe_allow_html=True)
            
            # Show Gauge Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pred,
                title = {'text': "AQI Level"},
                gauge = {'axis': {'range': [0, 500]}, 'bar': {'color': "#10b981"}}
            ))
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        # Show message if model hasn't been trained yet
        st.info(f"Please go to the 'Model' tab and train the model first. ({e})")

# --- TAB 6: GEOSPATIAL MAP ---
with tab6:
    st.header("Pollution Hotspots")
    
    # Define coordinates for major cities
    city_coords = {
        'Delhi':(28.61,77.20), 'Mumbai':(19.07,72.87), 'Bengaluru':(12.97,77.59),
        'Kolkata':(22.57, 88.36), 'Chennai':(13.08, 80.27), 'Hyderabad':(17.38, 78.48),
        'Ahmedabad':(23.02, 72.57), 'Lucknow':(26.84, 80.94)
    }
    
    # Group by City to get average AQI
    city_aqi = df.groupby('City')['AQI'].mean().round(0).reset_index()
    
    # Map coordinates to the dataframe
    city_aqi['lat'] = city_aqi['City'].map({k:v[0] for k,v in city_coords.items()})
    city_aqi['lon'] = city_aqi['City'].map({k:v[1] for k,v in city_coords.items()})
    
    # Plot Map
    fig = px.scatter_mapbox(
        city_aqi.dropna(), 
        lat="lat", lon="lon", 
        size="AQI", color="AQI",
        hover_name="City", zoom=3, 
        title="AQI Hotspots",
        color_continuous_scale="Reds"
    )
    fig.update_layout(mapbox_style="carto-positron", height=600)
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 7: ABOUT ---
with tab7:
    st.header("About This Project")
    st.markdown("""
    **CMP7005 – Programming for Data Analysis** **Student ID:** ST20316895  
    **Academic Year:** 2025–26  
    **Module Leader:** aprasad@cardiffmet.ac.uk  
    **Dataset:** India Air Quality (2015–2020)  
    **Built with:** Python • Pandas • Scikit-learn • Streamlit • Plotly  
    """)
    st.image("https://www.cardiffmet.ac.uk/PublishingImages/logo.png", width=300)
