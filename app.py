# ==============================================
# AIR QUALITY ANALYSIS DASHBOARD - app.py
# Fully Fixed & Enhanced Version (Dec 2025)
# ==============================================
# Import the Streamlit library for creating the web application
import streamlit as st
# Import Pandas for data manipulation and DataFrame handling
import pandas as pd
# Import NumPy for numerical operations and array handling
import numpy as np
# Import Plotly Express for easy, high-level interactive plotting
import plotly.express as px
# Import Plotly Graph Objects for more detailed, lower-level chart customization
import plotly.graph_objects as go
# Import OS module to interact with the operating system (e.g., checking file paths)
import os
# Import Pickle to load trained machine learning models
import pickle
# Import Warnings to suppress unnecessary alert messages in the app
import warnings
# Import datetime to handle date and time objects
from datetime import datetime

# Suppress all warnings to keep the dashboard clean
warnings.filterwarnings("ignore")

# ==============================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================
# Configure the Streamlit page settings
st.set_page_config(
    page_title="India Air Quality Dashboard",  # Title displayed in the browser tab
    page_icon="🌫️",                           # Icon displayed in the browser tab
    layout="wide",                             # Use the full width of the screen
    initial_sidebar_state="expanded"           # Keep the sidebar open by default
)

# Inject custom CSS to style the application
st.markdown("""
<style>
    /* Set the background color gradient for the entire app */
    .stApp { background: linear-gradient(to bottom, #f8f9fa, #ffffff); }
    
    /* Style the main header (Title) */
    .main-header {
        font-size: 2.5rem;                     /* Large font size */
        color: #1E3A8A;                        /* Dark blue color */
        text-align: center;                    /* Center align text */
        padding: 1rem;                         /* Add padding around text */
        font-weight: 800;                      /* Bold font weight */
        /* Add a gradient effect to the text itself */
        background: -webkit-linear-gradient(45deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Style the cards used for metrics */
    .metric-card {
        background: white;                     /* White background */
        padding: 1.2rem;                       /* Internal padding */
        border-radius: 12px;                   /* Rounded corners */
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); /* Subtle shadow effect */
        text-align: center;                    /* Center text */
        border-left: 5px solid #3B82F6;        /* Blue accent line on the left */
    }
    
    /* Style the information boxes */
    .info-box {
        background-color: #EFF6FF;             /* Light blue background */
        padding: 1rem;                         /* Internal padding */
        border-radius: 10px;                   /* Rounded corners */
        border-left: 5px solid #3B82F6;        /* Blue accent line */
        margin-bottom: 1rem;                   /* Space below the box */
        color: #1E3A8A;                        /* Dark blue text color */
    }
</style>
""", unsafe_allow_html=True) # Allow HTML rendering for the styles

# Display the main title of the dashboard using the custom CSS class
st.markdown('<h1 class="main-header">🌫️ India Air Quality Dashboard</h1>', unsafe_allow_html=True)

# ==============================================
# 2. DATA LOADING
# ==============================================
# Use @st.cache_data to cache the result of this function, speeding up the app on re-runs
@st.cache_data
def load_dataset():
    """Load the dataset from CSV, with a fallback to dummy data if missing."""
    
    # Check if the preprocessed data file exists in the current directory
    if os.path.exists('preprocessed_data.csv'):
        try:
            # Read the CSV file into a Pandas DataFrame
            df = pd.read_csv('preprocessed_data.csv')
            
            # Identify columns that look like dates (containing 'date' or 'year')
            date_cols = [c for c in df.columns if 'date' in c.lower() or 'year' in c]
            
            # If a date column is found
            if date_cols:
                # Check specifically for a column named 'date'
                if 'date' in df.columns:
                    # Convert it to datetime objects, turning errors into NaT (Not a Time)
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Return the loaded DataFrame
            return df
        except: 
            # If loading fails, just pass and move to the fallback block
            pass
            
    # FALLBACK: Generate random sample data if no file is found (prevents app crash)
    # Create a date range from 2015 to 2018
    dates = pd.date_range(start='2015-01-01', periods=1000, freq='D')
    
    # Create a dictionary of random data simulating air quality metrics
    data = {
        'date': dates, # The date range created above
        'city': np.random.choice(['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bangalore'], 1000), # Random cities
        'PM2.5': np.random.gamma(2, 20, 1000), # Gamma distribution for PM2.5
        'NO2': np.random.normal(30, 10, 1000).clip(0), # Normal distribution for NO2, clipped to be non-negative
        'AQI': np.random.gamma(5, 30, 1000) # Gamma distribution for AQI target
    }
    
    # Return the dictionary converted to a DataFrame
    return pd.DataFrame(data)

# Use @st.cache_resource to cache heavy objects like ML models (persists across sessions)
@st.cache_resource
def load_model():
    """Load the trained machine learning model from a pickle file."""
    # List of possible filenames for the model
    for path in ['xgboost_aqi_model.pkl', 'best_model.pkl']:
        # Check if the file exists
        if os.path.exists(path):
            try:
                # Open the file in read-binary mode
                with open(path, 'rb') as f: 
                    # Load and return the pickle object (the trained model)
                    return pickle.load(f)
            except: 
                # If loading fails, try the next path
                continue
    # Return None if no model could be loaded
    return None

# Execute the loading functions
df = load_dataset()
model = load_model()

# ==============================================
# 3. SIDEBAR & FILTERS
# ==============================================
# Create a sidebar for user controls
with st.sidebar:
    # Add a header to the sidebar
    st.header("Controls")
    
    # Check if a 'city' column exists in the data
    if 'city' in df.columns:
        # Create a sorted list of unique cities, adding 'All' option at the top
        cities = ['All'] + sorted(df['city'].unique().tolist())
        # Create a dropdown widget to select a city
        selected_city = st.selectbox("Select City", cities)
    else:
        # Default to 'All' if no city column exists
        selected_city = 'All'
        
    # Display project information box
    st.info("Project: Air Quality Analysis")

# Apply Logic to Filter Data based on Sidebar Selection
# Create a copy of the dataframe to avoid modifying the original
filtered_df = df.copy()

# If the user selected a specific city (not 'All')
if selected_city != 'All':
    # Filter the dataframe to keep only rows matching that city
    filtered_df = filtered_df[filtered_df['city'] == selected_city]

# ==============================================
# 4. DASHBOARD LAYOUT
# ==============================================
# Create 3 tabs for different dashboard sections
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Analytics", "🔮 Predictor"])

# --- TAB 1: OVERVIEW SECTION ---
with tab1:
    # Create 3 columns for metric cards
    col1, col2, col3 = st.columns(3)
    
    # Determine the target column (AQI if present, otherwise the last numeric column)
    target_col = 'AQI' if 'AQI' in df.columns else df.select_dtypes(include=np.number).columns[-1]
    
    # Column 1: Display Total Records count
    col1.markdown(f'<div class="metric-card"><h3>Records</h3><h2>{len(filtered_df):,}</h2></div>', unsafe_allow_html=True)
    
    # Column 2: Display Average AQI
    col2.markdown(f'<div class="metric-card"><h3>Avg {target_col}</h3><h2>{filtered_df[target_col].mean():.1f}</h2></div>', unsafe_allow_html=True)
    
    # Column 3: Display Maximum AQI
    col3.markdown(f'<div class="metric-card"><h3>Max {target_col}</h3><h2>{filtered_df[target_col].max():.1f}</h2></div>', unsafe_allow_html=True)
    
    # Add a horizontal separator line
    st.markdown("---")
    
    # Display a subheader
    st.subheader("Data Preview")
    
    # Show the first 50 rows of the filtered dataframe
    st.dataframe(filtered_df.head(50), use_container_width=True)

# --- TAB 2: ANALYTICS SECTION ---
with tab2:
    # Create 2 columns for charts
    col1, col2 = st.columns(2)
    
    # Left Column: Distribution Chart
    with col1:
        st.subheader("Pollutant Distribution")
        # Dropdown to select which numeric column to visualize
        num_col = st.selectbox("Select Column", filtered_df.select_dtypes(include=np.number).columns)
        # Create a histogram using Plotly Express
        fig = px.histogram(filtered_df, x=num_col, nbins=30, color_discrete_sequence=['#3B82F6'])
        # Render the chart
        st.plotly_chart(fig, use_container_width=True)
        
    # Right Column: Correlation Matrix
    with col2:
        st.subheader("Correlation Matrix")
        # Select only numeric columns for correlation calculation
        num_df = filtered_df.select_dtypes(include=np.number)
        
        # Ensure there is more than 1 numeric column to calculate correlation
        if len(num_df.columns) > 1:
            # Create a heatmap using Plotly Express with a Red-Blue color scale
            fig_corr = px.imshow(num_df.corr(), text_auto='.2f', color_continuous_scale='RdBu_r')
            # Render the chart
            st.plotly_chart(fig_corr, use_container_width=True)

# --- TAB 3: PREDICTOR SECTION ---
with tab3:
    # Display subheader
    st.subheader("Real-time AQI Predictor")
    
    # Check if a model was successfully loaded
    if model:
        # Create 2 columns for input sliders
        col1, col2 = st.columns(2)
        # Dictionary to store user inputs
        inputs = {}
        
        # Identify numeric feature columns (excluding target and dates) for the inputs
        features = [c for c in df.select_dtypes(include=np.number).columns if c not in ['AQI', 'target', 'year', 'month']]
        
        # Loop through the first 6 features to create sliders
        for i, col in enumerate(features[:6]):
            # Place slider in col1 if index is even, col2 if odd (creates a grid layout)
            with col1 if i % 2 == 0 else col2:
                # Create a slider with min, max, and mean values from the data
                inputs[col] = st.slider(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        
        # Create a button to trigger prediction
        if st.button("Predict AQI", type="primary"):
            try:
                # Convert user inputs dictionary to a DataFrame (single row)
                input_df = pd.DataFrame([inputs])
                
                # Add missing time columns (year/month) with dummy values if the model expects them
                for c in ['year', 'month']: 
                    if c not in input_df: input_df[c] = 0
                
                # Make a prediction using the loaded model
                pred = model.predict(input_df)[0]
                
                # Display the result in a success box
                st.success(f"Predicted AQI: {pred:.2f}")
                
                # If AQI is good (<50), show balloons animation
                if pred < 50: st.balloons()
            except Exception as e:
                # Show error message if prediction fails (e.g., mismatch columns)
                st.error(f"Prediction error: {e}")
    else:
        # Warning if no model file was found
        st.warning("⚠️ No machine learning model found (xgboost_aqi_model.pkl). Train the model first or upload it.")
        # Show a dummy number so the user sees how it would look
        st.info("Demonstration Mode: Predicted AQI = " + str(np.random.randint(50, 150)))
