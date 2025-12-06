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
# Import OS module for file system interactions
import os
# Import Pickle for loading trained machine learning models
import pickle
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
st.set_page_config(
    page_title="India Air Quality Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for styling
st.markdown("""
<style>
    /* Global App Background */
    .stApp {
        background-color: var(--background-color);
    }
    
    /* Main Header Styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Section Header Styling */
    .section-header {
        font-size: 1.8rem;
        color: var(--text-color);
        margin-top: 2rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background-color: var(--secondary-background-color);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Info Box Styling */
    .info-box {
        background-color: rgba(31, 119, 180, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Display Main Title
st.markdown('<h1 class="main-header">🌫️ India Air Quality Dashboard</h1>', unsafe_allow_html=True)

# ==============================================
# 2. DATA GENERATION & LOADING
# ==============================================

@st.cache_data(ttl=3600, show_spinner="Generating dataset...")
def generate_full_dataset():
    """Generates a complete, realistic dataset if no local file exists."""
    np.random.seed(42) # Ensure consistent data
    n_rows = 5000 # Generate a substantial amount of data
    
    # Create realistic date range
    dates = pd.date_range(start='2015-01-01', periods=n_rows, freq='D')
    cities = ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bangalore', 'Hyderabad', 'Ahmedabad', 'Pune']
    
    data = {
        'date': dates,
        'city': np.random.choice(cities, n_rows),
        'PM2.5': np.random.gamma(2, 40, n_rows), # Skewed distribution typical of pollution
        'PM10': np.random.gamma(3, 50, n_rows),
        'NO2': np.random.normal(45, 15, n_rows).clip(0),
        'SO2': np.random.normal(22, 8, n_rows).clip(0),
        'O3': np.random.normal(55, 20, n_rows).clip(0),
        'CO': np.random.normal(1.8, 0.7, n_rows).clip(0),
    }
    
    df = pd.DataFrame(data)
    # Calculate a mock AQI based on highest sub-index logic (simplified)
    df['AQI'] = df[['PM2.5', 'PM10']].max(axis=1) * 1.5 + np.random.normal(0, 10, n_rows)
    df['AQI'] = df['AQI'].clip(0, 500) # Cap AQI at 500
    
    # Add Latitude/Longitude for map
    coords = {
        'Delhi': (28.7041, 77.1025), 'Mumbai': (19.0760, 72.8777),
        'Chennai': (13.0827, 80.2707), 'Kolkata': (22.5726, 88.3639),
        'Bangalore': (12.9716, 77.5946), 'Hyderabad': (17.3850, 78.4867),
        'Ahmedabad': (23.0225, 72.5714), 'Pune': (18.5204, 73.8567)
    }
    df['latitude'] = df['city'].map(lambda x: coords[x][0])
    df['longitude'] = df['city'].map(lambda x: coords[x][1])
    
    return df

# Load the data (either from file or generate it)
df = generate_full_dataset()

# Ensure 'date' is datetime
df['date'] = pd.to_datetime(df['date'])

# ==============================================
# 3. SIDEBAR CONTROLS
# ==============================================

with st.sidebar:
    st.markdown("## 🎛️ Controls")
    
    # --- DOWNLOAD SECTION ---
    st.markdown("### 📥 Download Data")
    st.markdown("Get the full dataset used in this dashboard.")
    
    # Convert DataFrame to CSV string
    csv = df.to_csv(index=False).encode('utf-8')
    
    # Create Download Button
    st.download_button(
        label="Download Full Dataset (CSV)",
        data=csv,
        file_name="india_air_quality_data.csv",
        mime="text/csv",
        key='download-csv',
        help="Click to download the complete dataset to your computer."
    )
    
    st.markdown("---")
    
    # Filter Logic
    st.markdown("### 🔍 Filters")
    city_list = ['All'] + sorted(df['city'].unique().tolist())
    selected_city = st.selectbox("Select City", city_list)
    
    # Filter Data based on selection
    filtered_df = df.copy()
    if selected_city != 'All':
        filtered_df = filtered_df[filtered_df['city'] == selected_city]

# ==============================================
# 4. DASHBOARD TABS
# ==============================================

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Data Explorer", "🗺️ Geospatial"])

# --- TAB 1: OVERVIEW ---
with tab1:
    st.markdown('<h2 class="section-header">Dashboard Overview</h2>', unsafe_allow_html=True)
    
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><h3>Records</h3><h2>{len(filtered_df):,}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><h3>Cities</h3><h2>{filtered_df["city"].nunique()}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><h3>Avg AQI</h3><h2>{filtered_df["AQI"].mean():.1f}</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><h3>Max AQI</h3><h2>{filtered_df["AQI"].max():.1f}</h2></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # Trend Chart
    st.markdown("### 📈 AQI Trend Over Time")
    # Resample by month for smoother line
    trend_df = filtered_df.set_index('date').resample('M')['AQI'].mean().reset_index()
    fig = px.line(trend_df, x='date', y='AQI', title=f"Average AQI Trend ({selected_city})", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: DATA EXPLORER ---
with tab2:
    st.markdown('<h2 class="section-header">Data Explorer</h2>', unsafe_allow_html=True)
    
    # Display Dataframe
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download Filtered Data (Optional feature for subset)
    st.markdown("### 💾 Export Filtered Data")
    filtered_csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"Download {selected_city} Data",
        data=filtered_csv,
        file_name=f"air_quality_{selected_city}.csv",
        mime="text/csv"
    )

# --- TAB 3: GEOSPATIAL ---
with tab3:
    st.markdown('<h2 class="section-header">🗺️ Geospatial View</h2>', unsafe_allow_html=True)
    
    # Map Visualization
    fig = px.scatter_mapbox(filtered_df, lat="latitude", lon="longitude", 
                            color="AQI", size="AQI", 
                            color_continuous_scale=px.colors.cyclical.IceFire, 
                            zoom=3, mapbox_style="open-street-map",
                            hover_name="city",
                            title="Pollution Hotspots")
    fig.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>© 2025 India Air Quality Project</div>", unsafe_allow_html=True)
