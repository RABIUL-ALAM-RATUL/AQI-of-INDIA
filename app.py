# Import the Streamlit library for building the web application
import streamlit as st
# Import Pandas for data manipulation and DataFrame handling
import pandas as pd
# Import NumPy for numerical operations
import numpy as np
# Import Plotly Express for high-level interactive plotting
import plotly.express as px
# Import Plotly Graph Objects for detailed chart customization
import plotly.graph_objects as go
# Import Matplotlib for basic plotting if needed (mostly replaced by Plotly here)
import matplotlib.pyplot as plt
# Import OS module for file system interactions
import os
# Import Pickle for loading trained machine learning models
import pickle
# Import JSON for handling configuration files if needed
import json
# Import datetime for handling dates and times
from datetime import datetime
# Import warnings to suppress unnecessary console alerts
import warnings

# Suppress warnings to keep the application log clean
warnings.filterwarnings("ignore")

# ==============================================
# 1. PAGE CONFIGURATION & THEME ADAPTIVE CSS
# ==============================================

# Configure the Streamlit page settings (Title, Icon, Layout)
# This must be the first Streamlit command in the script
st.set_page_config(
    page_title="India Air Quality Dashboard",  # Title in browser tab
    page_icon="🌫️",                           # Icon in browser tab
    layout="wide",                             # Use full screen width
    initial_sidebar_state="expanded"           # Keep sidebar open by default
)

# Inject Custom CSS for styling
# Note: Using CSS variables (var(--...)) makes it adaptive to Streamlit's Light/Dark themes automatically
st.markdown("""
<style>
    /* Global App Background - uses theme's background color */
    .stApp {
        background-color: var(--background-color);
    }
    
    /* Main Header Styling */
    .main-header {
        font-size: 2.8rem;                     /* Large font size */
        font-weight: 800;                      /* Bold text */
        text-align: center;                    /* Center alignment */
        margin-bottom: 2rem;                   /* Space below header */
        /* Gradient text effect: Blue to Green */
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Section Header Styling (e.g., "Data Explorer") */
    .section-header {
        font-size: 1.8rem;                     /* Medium-large font */
        color: var(--text-color);              /* Theme-adaptive text color */
        margin-top: 2rem;                      /* Space above */
        border-bottom: 3px solid #1f77b4;      /* Blue underline */
        padding-bottom: 0.5rem;                /* Space between text and line */
    }
    
    /* Metric Card Styling (for KPI boxes) */
    .metric-card {
        background-color: var(--secondary-background-color); /* Theme-adaptive background */
        padding: 1.2rem;                       /* Internal padding */
        border-radius: 12px;                   /* Rounded corners */
        text-align: center;                    /* Center text */
        border-left: 5px solid #1f77b4;        /* Blue accent bar on left */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); /* Subtle shadow */
    }
    
    /* Info Box Styling (Blue) */
    .info-box {
        background-color: rgba(31, 119, 180, 0.1); /* Transparent Blue */
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    
    /* Success Box Styling (Green) */
    .success-box {
        background-color: rgba(44, 160, 44, 0.1); /* Transparent Green */
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2ca02c;
        margin-bottom: 1rem;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;                           /* Full width button */
        border-radius: 8px;                    /* Rounded corners */
        font-weight: bold;                     /* Bold text */
        background: linear-gradient(90deg, #1f77b4, #2ca02c); /* Gradient background */
        color: white;                          /* White text */
        border: none;                          /* No border */
        transition: transform 0.2s;            /* Smooth hover animation */
    }
    /* Button Hover Effect */
    .stButton>button:hover {
        transform: translateY(-2px);           /* Move up slightly */
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);/* Add shadow */
    }
</style>
""", unsafe_allow_html=True) # Allow HTML injection

# Display Main Title
st.markdown('<h1 class="main-header">🌫️ India Air Quality Dashboard</h1>', unsafe_allow_html=True)
# Display Intro Text Box
st.markdown("""
<div class="info-box">
    <strong>Comprehensive Analysis</strong> of air quality across Indian cities. 
    Explore trends, predict AQI, and gain insights interactively.
</div>
""", unsafe_allow_html=True)

# Initialize Session State for file uploads (persists across re-runs)
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None

# ==============================================
# 2. DATA LOADING & CACHING
# ==============================================

# Cache the data loading function to improve performance (TTL=1 hour)
@st.cache_data(ttl=3600, show_spinner="Loading data...")
def load_data():
    """Load dataset from session state, local file, or generate dummy data."""
    
    # Priority 1: Check if user uploaded a file in the current session
    if st.session_state.uploaded_df is not None:
        st.success("✅ Using uploaded dataset")
        return st.session_state.uploaded_df.copy()

    # Priority 2: Check for existing local files (generated from previous steps)
    data_paths = [
        'preprocessed_air_quality_data.csv',
        'data/preprocessed_air_quality_data.csv',
        'air_quality_data.csv'
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            try:
                # Read CSV
                df = pd.read_csv(path)
                # Standardize Date Column if present
                date_cols = [c for c in df.columns if 'date' in c.lower()]
                if date_cols:
                    df['date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
                st.success(f"✅ Loaded from: {path}")
                return df
            except: continue # If read fails, try next path

    # Priority 3: Fallback Dummy Data (prevents app crash during demo)
    st.warning("⚠️ No data found. Generating sample data for demonstration.")
    np.random.seed(42) # Set seed for reproducibility
    n = 1000 # Number of rows
    # Create DataFrame with simulated values
    df = pd.DataFrame({
        'city': np.random.choice(['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bangalore', 'Hyderabad'], n),
        'date': pd.date_range('2015-01-01', periods=n, freq='D'),
        'PM2.5': np.random.gamma(2, 30, n),
        'PM10': np.random.gamma(3, 40, n),
        'NO2': np.random.normal(45, 15, n).clip(0), # Clip to ensure non-negative
        'SO2': np.random.normal(22, 8, n).clip(0),
        'O3': np.random.normal(55, 20, n).clip(0),
        'CO': np.random.normal(1.8, 0.7, n).clip(0),
        'AQI': np.random.normal(160, 70, n).clip(0)
    })
    return df

# Cache model loading (stored in memory, not reloaded every interaction)
@st.cache_resource
def load_models():
    """Load machine learning models from .pkl files."""
    models = {}
    model_paths = ['.', 'models'] # Check current dir and 'models' subdir
    
    for directory in model_paths:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith('.pkl'): # Look for pickle files
                    try:
                        with open(os.path.join(directory, file), 'rb') as f:
                            # Create a readable name from filename (e.g., 'xgboost_model' -> 'Xgboost Model')
                            name = file.replace('.pkl', '').replace('_', ' ').title()
                            models[name] = pickle.load(f)
                    except: continue
    return models if models else None

# Execute Loading Functions
df = load_data()
models = load_models()

# Ensure 'date' column is datetime type for plotting
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

# ==============================================
# 3. HELPER FUNCTIONS
# ==============================================

def calculate_air_quality_category(aqi):
    """Return category label, color, and health message based on AQI value."""
    if aqi <= 50: return {"category": "Good", "color": "#00E400", "health": "Air quality is satisfactory"}
    elif aqi <= 100: return {"category": "Satisfactory", "color": "#FFFF00", "health": "Moderate for sensitive groups"}
    elif aqi <= 200: return {"category": "Moderate", "color": "#FF7E00", "health": "Unhealthy for sensitive groups"}
    elif aqi <= 300: return {"category": "Poor", "color": "#FF0000", "health": "Unhealthy"}
    elif aqi <= 400: return {"category": "Very Poor", "color": "#8F3F97", "health": "Very unhealthy"}
    else: return {"category": "Severe", "color": "#7E0023", "health": "Hazardous"}

# ==============================================
# 4. SIDEBAR CONTROLS
# ==============================================

with st.sidebar:
    st.markdown("## 🎛️ Controls") # Sidebar Header
    
    # Data Source Toggle
    data_source = st.radio("Data Source", ["Default", "Upload CSV/Excel"])
    
    # File Uploader Logic
    if data_source == "Upload CSV/Excel":
        uploaded_file = st.file_uploader("Choose file", type=['csv', 'xlsx', 'xls'])
        if uploaded_file:
            try:
                # Read file based on extension
                df_up = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.session_state.uploaded_df = df_up # Save to session state
                st.success(f"Uploaded: {uploaded_file.name}")
                st.rerun() # Refresh app to use new data
            except Exception as e:
                st.error(f"Error: {e}")

    # Model Selection Dropdown
    selected_model_name = st.selectbox("Prediction Model", 
        options=list(models.keys()) if models else ["Formula-Based"], 
        index=0)
    
    # Refresh Button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear() # Clear cache
        st.rerun() # Reload app

    # Filter Logic (Global Filters for the Sidebar)
    st.markdown("---")
    st.markdown("### 🔍 Filters")
    
    # City Filter
    if 'city' in df.columns:
        city_list = ['All'] + sorted(df['city'].unique().tolist())
        selected_city = st.selectbox("Select City", city_list)
    else:
        selected_city = 'All'
    
    # Filter Data based on selection
    filtered_df = df.copy()
    if selected_city != 'All':
        filtered_df = filtered_df[filtered_df['city'] == selected_city]

# ==============================================
# 5. DASHBOARD TABS
# ==============================================

# Create 6 Tabs for organized content
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🔍 Data Explorer", "📈 EDA", 
    "🤖 Models", "🔮 Predictor", "🗺️ Geospatial"
])

# --- TAB 1: OVERVIEW ---
with tab1:
    st.markdown('<h2 class="section-header">Dashboard Overview</h2>', unsafe_allow_html=True)
    
    # Create 4 columns for KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    # Determine target column (AQI or last numeric col)
    target_col = 'AQI' if 'AQI' in df.columns else df.select_dtypes(include=np.number).columns[-1]
    
    # Render KPIs using HTML cards for custom styling
    c1.markdown(f'<div class="metric-card"><h3>Records</h3><h2>{len(filtered_df):,}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><h3>Cities</h3><h2>{filtered_df["city"].nunique() if "city" in df.columns else "N/A"}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><h3>Avg {target_col}</h3><h2>{filtered_df[target_col].mean():.1f}</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><h3>Max {target_col}</h3><h2>{filtered_df[target_col].max():.1f}</h2></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # Overview Charts
    col1, col2 = st.columns([2,1]) # Left col wider (2/3), Right col narrower (1/3)
    
    with col1:
        st.markdown("### 📈 Recent Trends")
        # Ensure date exists for plotting
        if 'date' in filtered_df.columns:
            # Resample by month to make chart cleaner if too much data
            daily_df = filtered_df.set_index('date').resample('M')[target_col].mean().reset_index()
            fig = px.line(daily_df, x='date', y=target_col, title=f"{target_col} Trend Over Time", markers=True)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No date column found for trend analysis.")
            
    with col2:
        st.markdown("### 📊 Distribution")
        # Pie chart for AQI Categories
        if 'AQI' in df.columns:
            # Apply categorization function
            filtered_df['Category'] = filtered_df['AQI'].apply(lambda x: calculate_air_quality_category(x)['category'])
            # Create Pie Chart
            fig = px.pie(filtered_df, names='Category', title="AQI Category Distribution",
                         color='Category',
                         color_discrete_map={"Good":"#00E400","Satisfactory":"#FFFF00","Moderate":"#FF7E00",
                                             "Poor":"#FF0000","Very Poor":"#8F3F97","Severe":"#7E0023"})
            fig.update_layout(height=350, showlegend=False)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: DATA EXPLORER ---
with tab2:
    st.markdown('<h2 class="section-header">Data Explorer</h2>', unsafe_allow_html=True)
    
    # Interactive filtering options inside an expander
    with st.expander("🛠️ Advanced Filters", expanded=True):
        f1, f2 = st.columns(2)
        # Select numeric column to filter by range
        filter_col = f1.selectbox("Filter by Column", df.select_dtypes(include=np.number).columns)
        # Range slider for the selected column
        rng = f2.slider(f"{filter_col} Range", 
                        float(df[filter_col].min()), 
                        float(df[filter_col].max()), 
                        (float(df[filter_col].min()), float(df[filter_col].max())))
    
    # Apply range filter
    explorer_df = filtered_df[
        (filtered_df[filter_col] >= rng[0]) & 
        (filtered_df[filter_col] <= rng[1])
    ]
    
    # Sorting Controls
    s1, s2 = st.columns(2)
    sort_by = s1.selectbox("Sort By", explorer_df.columns)
    sort_asc = s2.radio("Order", ["Ascending", "Descending"]) == "Ascending"
    
    # Display Dataframe
    st.dataframe(explorer_df.sort_values(sort_by, ascending=sort_asc), use_container_width=True)
    
    # Display Summary Stats
    st.markdown("### 📋 Descriptive Statistics")
    st.dataframe(explorer_df.describe().T, use_container_width=True)

# --- TAB 3: EDA & INSIGHTS ---
with tab3:
    st.markdown('<h2 class="section-header">Deep Dive Analysis</h2>', unsafe_allow_html=True)
    
    # Dropdown to choose chart type
    analysis_type = st.selectbox("Choose Analysis Type", ["Univariate (Distribution)", "Bivariate (Scatter)", "Correlation Matrix"])
    
    if analysis_type == "Univariate (Distribution)":
        # User selects variable
        var = st.selectbox("Select Variable", df.select_dtypes(include=np.number).columns)
        c1, c2 = st.columns(2)
        # Histogram
        fig1 = px.histogram(filtered_df, x=var, nbins=30, title=f"Distribution of {var}", color_discrete_sequence=['#1f77b4'])
        c1.plotly_chart(fig1, use_container_width=True)
        # Box Plot
        fig2 = px.box(filtered_df, y=var, title=f"Box Plot of {var}", color_discrete_sequence=['#2ca02c'])
        c2.plotly_chart(fig2, use_container_width=True)
        
    elif analysis_type == "Bivariate (Scatter)":
        # User selects X and Y axes
        c1, c2 = st.columns(2)
        x_var = c1.selectbox("X Axis", df.select_dtypes(include=np.number).columns, index=0)
        y_var = c2.selectbox("Y Axis", df.select_dtypes(include=np.number).columns, index=1)
        # Scatter Plot
        fig = px.scatter(filtered_df, x=x_var, y=y_var, color=target_col, 
                         title=f"{x_var} vs {y_var} (Colored by {target_col})", 
                         trendline="ols") # Add regression line
        st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_type == "Correlation Matrix":
        # Calculate correlation
        corr = filtered_df.select_dtypes(include=np.number).corr()
        # Heatmap
        fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 4: MODEL INSIGHTS ---
with tab4:
    st.markdown('<h2 class="section-header">Model Performance</h2>', unsafe_allow_html=True)
    
    if models:
        st.success(f"✅ Loaded {len(models)} trained models.")
        
        # Determine available features (intersection of dataframe cols and typical features)
        # This prevents errors if data doesn't match training
        feature_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in ['AQI', 'target', 'year', 'month']]
        
        # Display Feature Importance (if model supports it)
        model = models[list(models.keys())[0]] # Pick first model
        if hasattr(model, 'feature_importances_'):
            st.subheader(f"Feature Importance ({list(models.keys())[0]})")
            # Create DataFrame for plotting
            fi_df = pd.DataFrame({
                'Feature': feature_cols[:len(model.feature_importances_)], # Ensure length match
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            # Bar Chart
            fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h', color='Importance')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("The selected model does not support feature importance visualization.")
    else:
        st.warning("⚠️ No trained models found. Please train a model in Step 5 and save it to the 'models' folder.")
        # Static placeholder visualization
        st.info("Demonstration Mode: Showing example model comparison.")
        demo_data = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
            'R2 Score': [0.65, 0.88, 0.92]
        })
        fig = px.bar(demo_data, x='Model', y='R2 Score', color='R2 Score', title="Model Accuracy Comparison")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 5: PREDICTOR ---
with tab5:
    st.markdown('<h2 class="section-header">🔮 AI AQI Predictor</h2>', unsafe_allow_html=True)
    
    # Input Area
    st.markdown("<div class='info-box'>Adjust the sliders below to simulate pollutant levels and predict Air Quality.</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Identify feature columns for inputs (exclude targets/dates)
    features = [c for c in df.select_dtypes(include=np.number).columns if c not in ['AQI', 'year', 'month', 'day']]
    
    # Dictionary to hold user inputs
    user_input = {}
    
    # Dynamically generate sliders for first 6 features
    for i, col in enumerate(features[:6]):
        # Distribute sliders across 3 columns
        with [col1, col2, col3][i % 3]:
            # Create slider based on min/max values in data
            user_input[col] = st.slider(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            
    st.markdown("---")
    
    # Prediction Action
    if st.button("🚀 Predict AQI", type="primary"):
        if models and selected_model_name in models:
            try:
                # Prepare input DataFrame
                input_df = pd.DataFrame([user_input])
                # Add dummy columns if model expects them (e.g., year/month)
                for c in ['year', 'month']:
                    if c not in input_df: input_df[c] = 0
                
                # Predict
                model = models[selected_model_name]
                pred = model.predict(input_df)[0]
                
                # Calculate category details
                cat = calculate_air_quality_category(pred)
                
                # Display Result Card
                st.markdown(f"""
                <div style="background-color: {cat['color']}; padding: 2rem; border-radius: 15px; text-align: center; color: white; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">
                    <h2 style="margin:0; color: white;">Predicted AQI</h2>
                    <h1 style="font-size: 4rem; margin: 0; color: white;">{pred:.0f}</h1>
                    <h3 style="margin:0; color: white;">{cat['category']}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Health Advice
                st.markdown(f"<div class='info-box' style='margin-top: 1rem;'><strong>🩺 Health Implication:</strong> {cat['health']}</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")
        else:
            # Fallback Formula Logic if no ML model
            # Simple average-based estimation for demo
            pred_val = np.mean(list(user_input.values())) * 1.5 
            st.info(f"Formula Estimation: {pred_val:.0f} (Load a model for better accuracy)")

# --- TAB 6: GEOSPATIAL ---
with tab6:
    st.markdown('<h2 class="section-header">🗺️ Geospatial Analysis</h2>', unsafe_allow_html=True)
    
    # Check if lat/lon columns exist
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Create Scatter Mapbox
        fig = px.scatter_mapbox(filtered_df, lat="latitude", lon="longitude", 
                                color=target_col, size=target_col, 
                                color_continuous_scale=px.colors.cyclical.IceFire, 
                                zoom=4, mapbox_style="open-street-map",
                                hover_name="city" if "city" in df.columns else None,
                                title=f"{target_col} Hotspots")
        fig.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No Latitude/Longitude columns found in the dataset.")
        
        # Simulated Map Data for Demonstration
        st.markdown("### 🌏 Simulated Map (Demo)")
        sim_data = pd.DataFrame({
            'lat': [28.61, 19.07, 13.08, 22.57, 12.97, 17.38],
            'lon': [77.20, 72.87, 80.27, 88.36, 77.59, 78.48],
            'city': ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bangalore', 'Hyderabad'],
            'AQI': [300, 150, 120, 200, 90, 110]
        })
        
        fig = px.scatter_mapbox(sim_data, lat="lat", lon="lon", color="AQI", size="AQI",
                                zoom=3, mapbox_style="open-street-map", 
                                hover_name="city",
                                color_continuous_scale='RdYlGn_r')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================
# FOOTER
# ==============================================
st.markdown("---")
# Centered footer text
st.markdown(f"""
<div style='text-align: center; padding: 2rem; color: #666; font-size: 0.9rem;'>
    <strong>India Air Quality Dashboard</strong> • Developed for Cardiff Met • {datetime.now().year}<br>
    Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
