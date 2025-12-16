# INDIA AIR QUALITY ANALYSIS APP (CMP7005 WRT1)
# DEVELOPED BY: MD RABIUL ALAM
# STUDENT ID: ST20316895

# -----------------------------------------------------------------------------
# 1. LIBRARY IMPORTS
# -----------------------------------------------------------------------------
import streamlit as st                  # Web App Framework
import pandas as pd                     # Data Manipulation
import numpy as np                      # Numerical Operations
import plotly.express as px             # Interactive Charts (High Level)
import plotly.graph_objects as go       # Interactive Charts (Detailed)
from sklearn.model_selection import train_test_split # Split Data
from sklearn.ensemble import RandomForestRegressor   # ML Model
from sklearn.metrics import r2_score    # Model Evaluation Metric
import joblib                           # Save/Load Models
import warnings                         # Warning Management

# Suppress warnings to keep the UI clean
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 2. PAGE CONFIGURATION & CSS STYLING
# -----------------------------------------------------------------------------
st.set_page_config(page_title="India Air Quality", layout="wide", page_icon="🌤️")

# Custom CSS for a professional, dark-themed look
st.markdown("""
<style>
    /* Header Container Style */
    .header {
        background: linear-gradient(90deg, #001f3f, #003366);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 30px;
    }
    
    /* Header Text Styles */
    .title { font-size: 50px; font-weight: bold; margin: 0; }
    .subtitle { font-size: 24px; margin-top: 10px; color: #e2e8f0; }
    
    /* Metric Card Style (KPI Boxes) */
    .metric-card {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* KPI Number Style */
    .metric-card h2 { margin: 10px 0 0 0; color: #38bdf8; }
    
    /* Tab Styling */
    .stTabs [data-testid="stTab"] {
        font-weight: bold;
        font-size: 16px;
    }
</style>

<div class="header">
    <div class="title">India Air Quality Analysis</div>
    <div class="subtitle">CMP7005 • ST20316895 • 2025-26</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. DATA LOADING (ROBUST)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # LOAD THE PROCESSED (SCALED) FILE
        # This file contains Z-scores for pollutants (mean=0, std=1)
        # and likely contains the raw AQI target.
        df = pd.read_csv("India_Air_Quality_Final_Processed.csv")

        # Standardize column names (remove spaces)
        df.columns = [c.strip() for c in df.columns]

        # Ensure Date column is in datetime format
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Drop rows where the Target (AQI) is missing
        # We cannot train or display trends if the main variable is NaN
        if 'AQI' in df.columns:
            df = df.dropna(subset=['AQI'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}. Please check if 'India_Air_Quality_Final_Processed.csv' exists.")
        return pd.DataFrame()

# Execute Load
df = load_data()

# Stop app if data failed to load
if df.empty:
    st.stop()

# -----------------------------------------------------------------------------
# 4. MAIN APP TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Home", "EDA", "Seasonal", "Model", "Predict", "Map", "About"
])

# --- TAB 1: DASHBOARD ---
with tab1:
    st.header("Project Dashboard")
    
    # Calculate KPIs
    # Note: AQI is unscaled in your target column, so these numbers make sense (e.g., 150, 200).
    total_records = len(df)
    total_cities = df['City'].nunique() if 'City' in df.columns else 0
    avg_aqi = df['AQI'].mean()
    max_aqi = df['AQI'].max()

    # Display KPIs in 4 columns
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card">Total Records<br><h2>{total_records:,}</h2></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card">Cities Covered<br><h2>{total_cities}</h2></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card">Average AQI<br><h2>{avg_aqi:.1f}</h2></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card">Max Recorded AQI<br><h2>{max_aqi:.0f}</h2></div>', unsafe_allow_html=True)

    # Trend Chart
    st.markdown("### National AQI Trends")
    # We sample data to prevent the chart from being too slow
    sample_data = df.sample(min(5000, len(df))) if len(df) > 5000 else df
    
    if 'Date' in df.columns and 'City' in df.columns:
        fig = px.line(sample_data, x='Date', y='AQI', color='City', title="AQI Over Time (Sampled)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Date or City column missing for trend analysis.")

# --- TAB 2: EDA (Exploratory Data Analysis) ---
with tab2:
    st.header("Correlation Analysis")
    
    # Select numeric columns for correlation
    # We drop AQI to see how *features* correlate with each other
    numeric_df = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
    
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Pollutant Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough numeric data for correlation.")

# --- TAB 3: SEASONAL PATTERNS ---
with tab3:
    st.header("Seasonal Analysis")
    
    if 'Date' in df.columns:
        # Create a copy to avoid SettingWithCopy warnings
        df_season = df.copy()
        
        # Extract Season from Month
        df_season['Season'] = df_season['Date'].dt.month.map({
            12:'Winter', 1:'Winter', 2:'Winter',
            3:'Spring', 4:'Spring', 5:'Spring',
            6:'Summer', 7:'Summer', 8:'Summer',
            9:'Monsoon', 10:'Monsoon', 11:'Monsoon'
        })
        
        # Box Plot
        st.markdown("### AQI Distribution by Season")
        fig = px.box(df_season, x='Season', y='AQI', color='Season', 
                     title="Seasonal Air Quality Levels",
                     category_orders={"Season": ["Winter", "Spring", "Summer", "Monsoon"]})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Date column required for seasonal analysis.")

# --- TAB 4: MODEL TRAINING ---
with tab4:
    st.header("Train Machine Learning Model")
    st.markdown("""
    This section trains a **Random Forest Regressor**.
    * **Inputs (X):** Scaled Pollutants (PM2.5, NO2, etc. as Z-scores)
    * **Target (y):** AQI (Original scale 0-500)
    """)

    # Prepare Data
    # X = All numeric columns EXCEPT AQI
    X = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
    # y = AQI
    y = df['AQI']

    # Train Button
    if st.button("🚀 Train Model Now", type="primary"):
        with st.spinner("Training model... (This may take a moment)"):
            
            # 1. Split Data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 2. Initialize Model
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            
            # 3. Fit Model
            model.fit(X_train, y_train)
            
            # 4. Evaluate
            pred = model.predict(X_test)
            r2 = r2_score(y_test, pred)
            
            # 5. Save Model
            joblib.dump(model, "aqi_model.pkl")
            
            # 6. Success Message
            st.success(f"Training Complete! Model Accuracy (R² Score): {r2:.4f}")
            st.balloons()

# --- TAB 5: LIVE PREDICTION (ADJUSTED FOR SCALED DATA) ---
with tab5:
    st.header("Predict Air Quality")
    
    try:
        # Load the model
        model = joblib.load("aqi_model.pkl")
        
        # Identify features expected by the model
        # These will match the columns in your Processed CSV
        X_feats = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
        feature_names = X_feats.columns.tolist()

        st.info("ℹ️ **NOTE:** The input data is **Standardized (Z-Scores)**. \n\n"
                "• **0.0** = Average Pollution Level\n"
                "• **+2.0** = Very High Pollution\n"
                "• **-2.0** = Very Low Pollution")

        # Create input sliders
        inputs = []
        
        # Use columns to organize sliders neatly
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        # Only create sliders for the top 9 features to avoid clutter
        # We assume the user wants to adjust the main pollutants
        display_features = feature_names[:9]

        for i, f in enumerate(display_features):
            with cols[i % 3]:
                # SLIDER RANGE ADJUSTED FOR SCALED DATA (-5 to +5)
                # Default is 0.0 (Average)
                val = st.slider(f"{f} (Z-Score)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
                inputs.append(val)
        
        # Button to Predict
        if st.button("Predict AQI Level", type="primary", use_container_width=True):
            
            # Handle Remaining Features
            # If the model expects more features than we displayed sliders for,
            # we fill them with 0.0 (the mean of scaled data).
            remaining_count = len(feature_names) - len(inputs)
            full_input = inputs + [0.0] * remaining_count
            
            # Reshape for Sklearn (1 row, N columns)
            input_array = np.array([full_input])
            
            # Make Prediction
            prediction = model.predict(input_array)[0]
            
            # Display Result
            st.divider()
            
            # Create two columns for the result display
            res_c1, res_c2 = st.columns([1, 2])
            
            with res_c1:
                st.markdown("### Predicted AQI")
                # Large Green Number
                st.markdown(f"<h1 style='color:#10b981; font-size: 70px; margin:0;'>{prediction:.0f}</h1>", unsafe_allow_html=True)
            
            with res_c2:
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction,
                    title = {'text': "Severity Scale"},
                    gauge = {
                        'axis': {'range': [0, 500]},
                        'bar': {'color': "white"},
                        'steps': [
                            {'range': [0, 100], 'color': "#00b894"},   # Good/Sat (Green)
                            {'range': [100, 200], 'color': "#fdcb6e"},  # Moderate (Yellow)
                            {'range': [200, 300], 'color': "#e17055"},  # Poor (Orange)
                            {'range': [300, 500], 'color': "#d63031"}   # Severe (Red)
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': prediction
                        }
                    }
                ))
                fig.update_layout(height=250, margin=dict(t=30, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.warning("⚠️ Model not found. Please go to the **'Model'** tab and click 'Train Model Now'.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- TAB 6: GEOSPATIAL MAP ---
with tab6:
    st.header("Pollution Hotspots")
    
    if 'City' in df.columns and 'AQI' in df.columns:
        # Define Coordinates for major Indian cities manually
        # (Since the dataset likely doesn't have Lat/Lon columns)
        city_coords = {
            'Delhi': (28.61, 77.20), 'Mumbai': (19.07, 72.87), 'Bengaluru': (12.97, 77.59),
            'Kolkata': (22.57, 88.36), 'Chennai': (13.08, 80.27), 'Hyderabad': (17.38, 78.48),
            'Ahmedabad': (23.02, 72.57), 'Lucknow': (26.84, 80.94), 'Patna': (25.59, 85.13),
            'Gurugram': (28.45, 77.02), 'Amritsar': (31.63, 74.87), 'Jaipur': (26.91, 75.78)
        }
        
        # Aggregate AQI by City
        city_stats = df.groupby('City')['AQI'].mean().reset_index()
        
        # Map coordinates
        city_stats['lat'] = city_stats['City'].map(lambda x: city_coords.get(x, (None, None))[0])
        city_stats['lon'] = city_stats['City'].map(lambda x: city_coords.get(x, (None, None))[1])
        
        # Remove cities where we don't have coordinates
        map_data = city_stats.dropna(subset=['lat', 'lon'])
        
        if not map_data.empty:
            fig = px.scatter_mapbox(
                map_data, lat="lat", lon="lon",
                size="AQI", color="AQI",
                hover_name="City",
                zoom=3.5,
                color_continuous_scale="RdYlGn_r", # Red = Bad, Green = Good
                title="Average AQI by City"
            )
            fig.update_layout(mapbox_style="carto-positron", height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not map cities. Coordinate data missing.")
    else:
        st.error("City or AQI column missing.")

# --- TAB 7: ABOUT ---
with tab7:
    st.header("About This Project")
    st.markdown("""
    ### India Air Quality Analysis App
    
    This application is part of the **CMP7005 Data Analysis** module assessment.
    
    **Developed By:**
    * **Name:** MD RABIUL ALAM
    * **Student ID:** ST20316895
    
    **Features:**
    * **Data Processing:** Handles standardized/scaled pollutant data.
    * **Visualization:** Interactive trends, correlations, and maps using Plotly.
    * **Machine Learning:** Random Forest Regressor trained live on the processed dataset.
    * **Prediction:** Real-time AQI inference based on pollutant Z-scores.
    
    **Tools Used:** Python, Streamlit, Pandas, Scikit-Learn, Plotly.
    """)
