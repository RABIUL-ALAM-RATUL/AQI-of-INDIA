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
    .title { font-size: 50px; font-weight: bold; margin: 0; }
    .subtitle { font-size: 24px; margin-top: 10px; color: #e2e8f0; }
    
    /* Metric Card Style */
    .metric-card {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        border: 1px solid #334155;
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

# -----------------------------------------------------------------------------
# 3. DATA LOADING (SUPER ROBUST)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # Load the CSV
        df = pd.read_csv("India_Air_Quality_Final_Processed.csv")

        # 1. Clean Column Headers (remove spaces)
        df.columns = [c.strip() for c in df.columns]

        # 2. INTELLIGENT RENAMING
        # This maps any variation of "city", "date", "aqi" to the standard names
        rename_map = {
            'city': 'City', 'city_name': 'City', 'place': 'City',
            'date': 'Date', 'dt': 'Date',
            'aqi': 'AQI', 'aqi_value': 'AQI',
            'aqi_bucket': 'AQI_Bucket'
        }
        
        # Create a dictionary for renaming by checking lowercase versions
        actual_rename = {}
        for col in df.columns:
            if col.lower() in rename_map:
                actual_rename[col] = rename_map[col.lower()]
        
        df.rename(columns=actual_rename, inplace=True)

        # 3. FIX CITY COLUMN (The "1 City" Fix)
        # Ensure City is a string and remove whitespace (e.g. "Delhi " -> "Delhi")
        if 'City' in df.columns:
            df['City'] = df['City'].astype(str).str.strip()

        # 4. FIX DATE COLUMN
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # 5. FILTER: Keep rows with valid Target (AQI)
        # We perform a check before dropping to avoid emptying the whole dataset
        if 'AQI' in df.columns:
            initial_count = len(df)
            df = df.dropna(subset=['AQI'])
            dropped = initial_count - len(df)
            if dropped > 0 and len(df) == 0:
                st.error("⚠️ CRITICAL: All rows were dropped because 'AQI' is missing. Please check your CSV.")
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load Data
df = load_data()

# Stop if failed
if df.empty:
    st.warning("Dataframe is empty. Please check the 'India_Air_Quality_Final_Processed.csv' file.")
    st.stop()

# --- SIDEBAR DIAGNOSTICS (Hidden by default, useful for debugging) ---
with st.sidebar:
    st.header("⚙️ Settings & Info")
    if st.checkbox("Show Dataset Info (Debug)"):
        st.write("**Detected Columns:**", df.columns.tolist())
        if 'City' in df.columns:
            cities = df['City'].unique()
            st.write(f"**Cities Found ({len(cities)}):**", cities)
        else:
            st.error("❌ 'City' column MISSING!")

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
    # Use robust checks to avoid errors if columns are missing
    total_records = len(df)
    
    if 'City' in df.columns:
        total_cities = df['City'].nunique()
    else:
        total_cities = 0
        st.warning("⚠️ 'City' column not detected. Check the Debug sidebar.")

    avg_aqi = df['AQI'].mean() if 'AQI' in df.columns else 0
    max_aqi = df['AQI'].max() if 'AQI' in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card">Total Records<br><h2>{total_records:,}</h2></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card">Cities Covered<br><h2>{total_cities}</h2></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card">Average AQI<br><h2>{avg_aqi:.1f}</h2></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card">Max Recorded AQI<br><h2>{max_aqi:.0f}</h2></div>', unsafe_allow_html=True)

    st.markdown("### National AQI Trends")
    if 'Date' in df.columns and 'City' in df.columns and 'AQI' in df.columns:
        sample_data = df.sample(min(5000, len(df))) if len(df) > 5000 else df
        fig = px.line(sample_data, x='Date', y='AQI', color='City', title="AQI Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Chart requires Date, City, and AQI columns.")

# --- TAB 2: EDA (Adaptive Heatmap) ---
with tab2:
    st.header("Correlation Analysis")
    
    numeric_df = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
    
    if not numeric_df.empty:
        corr = numeric_df.corr()
        # FIXED: Adaptive height and aspect ratio
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', 
                        title="Pollutant Correlation Matrix", height=700, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough numeric data for correlation.")

# --- TAB 3: SEASONAL PATTERNS ---
with tab3:
    st.header("Seasonal Analysis")
    
    if 'Date' in df.columns and 'AQI' in df.columns:
        df_season = df.copy()
        df_season['Season'] = df_season['Date'].dt.month.map({
            12:'Winter', 1:'Winter', 2:'Winter',
            3:'Spring', 4:'Spring', 5:'Spring',
            6:'Summer', 7:'Summer', 8:'Summer',
            9:'Monsoon', 10:'Monsoon', 11:'Monsoon'
        })
        
        fig = px.box(df_season, x='Season', y='AQI', color='Season', 
                     title="Seasonal Air Quality Levels",
                     category_orders={"Season": ["Winter", "Spring", "Summer", "Monsoon"]})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Date or AQI column missing for seasonal analysis.")

# --- TAB 4: MODEL TRAINING ---
with tab4:
    st.header("Train Machine Learning Model")
    st.markdown("Trains a Random Forest Regressor on Scaled Pollutants.")

    if 'AQI' not in df.columns:
        st.error("Target 'AQI' column missing. Cannot train model.")
    else:
        X = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
        y = df['AQI']

        if st.button("🚀 Train Model Now", type="primary"):
            with st.spinner("Training..."):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                r2 = r2_score(y_test, pred)
                joblib.dump(model, "aqi_model.pkl")
                st.success(f"Training Complete! R² Score: {r2:.4f}")
                st.balloons()

# --- TAB 5: PREDICT (Scaled Inputs) ---
with tab5:
    st.header("Predict Air Quality")
    
    try:
        model = joblib.load("aqi_model.pkl")
        X_feats = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
        feature_names = X_feats.columns.tolist()

        st.info("ℹ️ Input Data: Standardized (Z-Scores). 0.0 = Average.")

        inputs = []
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        for i, f in enumerate(feature_names[:9]):
            with cols[i % 3]:
                # Range -5 to +5 for Scaled Data
                val = st.slider(f"{f} (Z-Score)", -5.0, 5.0, 0.0, 0.1)
                inputs.append(val)
        
        if st.button("Predict AQI", type="primary", use_container_width=True):
            full_input = inputs + [0.0] * (len(feature_names) - len(inputs))
            input_array = np.array([full_input])
            prediction = model.predict(input_array)[0]
            
            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("### Prediction")
                st.markdown(f"<h1 style='color:#10b981; font-size:60px;'>{prediction:.0f}</h1>", unsafe_allow_html=True)
            with c2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prediction, title={'text': "Severity"},
                    gauge={'axis': {'range': [0, 500]}, 'bar': {'color': "white"},
                           'steps': [{'range': [0, 100], 'color': "#00b894"}, {'range': [100, 200], 'color': "#fdcb6e"},
                                     {'range': [200, 500], 'color': "#d63031"}]}
                ))
                fig.update_layout(height=250, margin=dict(t=30, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.warning("⚠️ Model not found. Train it in the 'Model' tab.")
    except Exception as e:
        st.error(f"Prediction Error: {e}")

# --- TAB 6: MAP (Robust Fix) ---
with tab6:
    st.header("Pollution Hotspots")
    
    # Check for BOTH columns explicitly
    has_city = 'City' in df.columns
    has_aqi = 'AQI' in df.columns
    
    if has_city and has_aqi:
        city_coords = {
            'Delhi': (28.61, 77.20), 'Mumbai': (19.07, 72.87), 'Bengaluru': (12.97, 77.59),
            'Kolkata': (22.57, 88.36), 'Chennai': (13.08, 80.27), 'Hyderabad': (17.38, 78.48),
            'Ahmedabad': (23.02, 72.57), 'Lucknow': (26.84, 80.94), 'Patna': (25.59, 85.13),
            'Gurugram': (28.45, 77.02), 'Amritsar': (31.63, 74.87), 'Jaipur': (26.91, 75.78),
            'Visakhapatnam': (17.68, 83.21), 'Thiruvananthapuram': (8.52, 76.93), 'Nagpur': (21.14, 79.08),
            'Chandigarh': (30.73, 76.77), 'Bhopal': (23.25, 77.41), 'Shillong': (25.57, 91.89)
        }
        
        # Calculate stats
        city_stats = df.groupby('City')['AQI'].mean().reset_index()
        
        # Map lat/lon
        city_stats['lat'] = city_stats['City'].map(lambda x: city_coords.get(x, (None, None))[0])
        city_stats['lon'] = city_stats['City'].map(lambda x: city_coords.get(x, (None, None))[1])
        
        # Drop unmapped cities
        map_data = city_stats.dropna(subset=['lat', 'lon'])
        
        if not map_data.empty:
            fig = px.scatter_mapbox(
                map_data, lat="lat", lon="lon", size="AQI", color="AQI",
                hover_name="City", zoom=3.5, color_continuous_scale="RdYlGn_r",
                title="Average AQI by City"
            )
            fig.update_layout(mapbox_style="carto-positron", height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not map any cities. The city names in your CSV might not match the coordinates list (e.g., 'Delhi' vs 'New Delhi').")
            st.write("Cities found in data:", df['City'].unique())
    else:
        st.error(f"Cannot render map. Missing Columns: {'City ' if not has_city else ''}{'AQI' if not has_aqi else ''}")

# --- TAB 7: ABOUT ---
with tab7:
    st.header("About This Project")
    st.markdown("""
    **CMP7005 Data Analysis Assessment** | **Student:** ST20316895
    * **Data:** Scaled India Air Quality (2015-2020)
    * **Tech:** Python, Streamlit, Scikit-Learn, Plotly
    """)
