# ==============================================
# AIR QUALITY ANALYSIS DASHBOARD - app.py
# ==============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import pickle
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Air Quality Analysis Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f0f7ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff7e6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff9900;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e6f7e6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2ca02c;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2ca02c, #1f77b4);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stSelectbox, .stSlider, .stNumberInput {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<h1 class="main-header">🌫️ Air Quality Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <strong>📊 Comprehensive Dashboard</strong> for analyzing air quality data, exploring patterns, 
    and predicting Air Quality Index (AQI) using machine learning models.
</div>
""", unsafe_allow_html=True)

# ==============================================
# DATA LOADING & CACHING FUNCTIONS
# ==============================================

@st.cache_data(ttl=3600)
def load_data():
    """Load preprocessed data"""
    try:
        # Try to load from different paths
        data_paths = [
            'preprocessed_air_quality_data.csv',
            'data/preprocessed_air_quality_data.csv',
            'air_quality_data.csv',
            'data/air_quality_data.csv'
        ]
        
        df = None
        for path in data_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.success(f"✅ Data loaded from: {path}")
                break
        
        if df is None:
            st.error("❌ Could not find data file. Using sample data.")
            # Create sample data for demonstration
            np.random.seed(42)
            n_samples = 1000
            df = pd.DataFrame({
                'city': np.random.choice(['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bangalore'], n_samples),
                'date': pd.date_range('2015-01-01', periods=n_samples, freq='D'),
                'PM2_5': np.random.normal(100, 30, n_samples).clip(0),
                'PM10': np.random.normal(150, 40, n_samples).clip(0),
                'NO2': np.random.normal(40, 10, n_samples).clip(0),
                'SO2': np.random.normal(20, 5, n_samples).clip(0),
                'O3': np.random.normal(50, 15, n_samples).clip(0),
                'CO': np.random.normal(1.5, 0.5, n_samples).clip(0),
                'AQI': np.random.normal(150, 50, n_samples).clip(0)
            })
            st.warning("⚠ Using generated sample data. Please upload your actual data.")
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        models = {}
        
        # Check for model files
        model_dir = 'models'
        if not os.path.exists(model_dir):
            st.warning("⚠ Models directory not found. Will use default models.")
            return None
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        if not model_files:
            st.warning("⚠ No trained models found. Will use default models.")
            return None
        
        st.success(f"✅ Found {len(model_files)} model(s) in {model_dir}")
        
        # Load each model
        for model_file in model_files[:5]:  # Load first 5 models
            try:
                model_path = os.path.join(model_dir, model_file)
                with open(model_path, 'rb') as f:
                    model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
                    models[model_name] = pickle.load(f)
                    st.success(f"  ✓ Loaded: {model_name}")
            except Exception as e:
                st.warning(f"  ⚠ Could not load {model_file}: {str(e)}")
        
        return models
    
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None

@st.cache_data
def load_model_config():
    """Load model configuration"""
    try:
        config_paths = [
            'models/model_config.json',
            'model_config.json'
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    config = json.load(f)
                    return config
        
        return None
    
    except Exception as e:
        st.warning(f"⚠ Could not load model config: {str(e)}")
        return None

# ==============================================
# HELPER FUNCTIONS
# ==============================================

def create_info_box(text, type="info"):
    """Create an information box"""
    if type == "info":
        st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)
    elif type == "warning":
        st.markdown(f'<div class="warning-box">{text}</div>', unsafe_allow_html=True)
    elif type == "success":
        st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)

def calculate_air_quality_category(aqi_value):
    """Calculate air quality category based on AQI value"""
    if aqi_value <= 50:
        return {"category": "Good", "color": "#00E400", "health_impact": "Minimal impact"}
    elif aqi_value <= 100:
        return {"category": "Satisfactory", "color": "#FFFF00", "health_impact": "Minor breathing discomfort to sensitive people"}
    elif aqi_value <= 200:
        return {"category": "Moderate", "color": "#FF7E00", "health_impact": "Breathing discomfort to people with lung disease"}
    elif aqi_value <= 300:
        return {"category": "Poor", "color": "#FF0000", "health_impact": "Breathing discomfort to most people on prolonged exposure"}
    elif aqi_value <= 400:
        return {"category": "Very Poor", "color": "#8F3F97", "health_impact": "Respiratory illness on prolonged exposure"}
    else:
        return {"category": "Severe", "color": "#7E0023", "health_impact": "Affects healthy people and seriously impacts those with existing diseases"}

# ==============================================
# SIDEBAR CONFIGURATION
# ==============================================

with st.sidebar:
    st.markdown("## 🎛️ Dashboard Controls")
    
    # Data source selection
    st.markdown("### 📁 Data Source")
    data_source = st.radio(
        "Select data source:",
        ["Preprocessed Data", "Upload New Data"],
        index=0
    )
    
    if data_source == "Upload New Data":
        uploaded_file = st.file_uploader(
            "Upload CSV file", 
            type=['csv', 'xlsx', 'xls'],
            help="Upload your air quality data file"
        )
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.success(f"✅ Uploaded: {uploaded_file.name} ({len(df)} rows)")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    # Model selection
    st.markdown("### 🤖 Model Selection")
    models = load_models()
    if models:
        selected_model = st.selectbox(
            "Choose prediction model:",
            list(models.keys())
        )
    else:
        selected_model = "Default Model"
        st.warning("Using default model")
    
    # Analysis timeframe
    st.markdown("### 📅 Time Range")
    timeframe = st.selectbox(
        "Select analysis timeframe:",
        ["All Time", "Last Year", "Last 6 Months", "Last Month", "Custom Range"],
        index=0
    )
    
    # City selection
    st.markdown("### 🏙️ City Selection")
    cities = st.multiselect(
        "Select cities to analyze:",
        ["All Cities", "Delhi", "Mumbai", "Chennai", "Kolkata", "Bangalore", "Hyderabad"],
        default=["All Cities"]
    )
    
    # Pollutant thresholds
    st.markdown("### ⚠️ Pollutant Thresholds")
    pm25_threshold = st.slider("PM2.5 Threshold (µg/m³)", 0, 500, 60, help="WHO guideline: 25 µg/m³")
    pm10_threshold = st.slider("PM10 Threshold (µg/m³)", 0, 500, 100, help="WHO guideline: 50 µg/m³")
    
    # Refresh button
    st.markdown("---")
    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.rerun()
    
    # Information
    st.markdown("---")
    with st.expander("ℹ️ About this Dashboard"):
        st.markdown("""
        **Air Quality Analysis Dashboard**
        
        This dashboard provides comprehensive analysis of air quality data including:
        
        • **Data Overview**: Summary statistics and data quality
        • **Exploratory Analysis**: Interactive visualizations and insights
        • **Model Insights**: Machine learning model performance
        • **AQI Predictor**: Real-time air quality predictions
        
        **Data Sources**: Indian air quality monitoring stations
        **Time Period**: 2015-2020
        **Models**: Multiple ML algorithms for prediction
        """)

# ==============================================
# MAIN DASHBOARD LAYOUT
# ==============================================

# Load data
df = load_data()

if df is None:
    st.error("❌ Failed to load data. Please check data source.")
    st.stop()

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard Overview",
    "🔍 Data Explorer",
    "📈 EDA & Insights",
    "🤖 Model Insights",
    "🔮 AQI Predictor",
    "🗺️ Geospatial View"
])

# ==============================================
# TAB 1: DASHBOARD OVERVIEW
# ==============================================

with tab1:
    st.markdown('<h2 class="section-header">📊 Dashboard Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_samples = len(df)
        st.metric("Total Samples", f"{total_samples:,}", "Data Points")
    
    with col2:
        total_cities = df['city'].nunique() if 'city' in df.columns else "N/A"
        st.metric("Cities Covered", total_cities, "Locations")
    
    with col3:
        if 'date' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['date']):
                date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
            else:
                date_range = f"{df['date'].min()} to {df['date'].max()}"
            st.metric("Date Range", date_range, "Time Period")
        else:
            st.metric("Date Range", "N/A", "Check data")
    
    with col4:
        if 'AQI' in df.columns:
            avg_aqi = df['AQI'].mean()
            aqi_category = calculate_air_quality_category(avg_aqi)['category']
            st.metric("Average AQI", f"{avg_aqi:.1f}", aqi_category)
    
    # Divider
    st.markdown("---")
    
    # Recent data preview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Recent Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Data quality metrics
        st.markdown("### ✅ Data Quality Check")
        quality_col1, quality_col2, quality_col3 = st.columns(3)
        
        with quality_col1:
            missing_percent = (df.isnull().sum().sum() / df.size) * 100
            st.metric("Missing Data", f"{missing_percent:.1f}%", 
                     delta="Good" if missing_percent < 5 else "Needs attention",
                     delta_color="normal" if missing_percent < 5 else "inverse")
        
        with quality_col2:
            duplicate_percent = (df.duplicated().sum() / len(df)) * 100
            st.metric("Duplicate Rows", f"{duplicate_percent:.1f}%", 
                     delta="Clean" if duplicate_percent < 1 else "Check needed",
                     delta_color="normal" if duplicate_percent < 1 else "inverse")
        
        with quality_col3:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Features", numeric_cols, 
                     delta=f"{len(df.columns)} total columns")
    
    with col2:
        st.markdown("### 📈 Quick Statistics")
        
        # Select a pollutant to view statistics
        pollutant_options = [col for col in ['PM2_5', 'PM10', 'NO2', 'SO2', 'O3', 'CO', 'AQI'] 
                           if col in df.columns]
        
        if pollutant_options:
            selected_pollutant = st.selectbox("Select Pollutant", pollutant_options)
            
            if selected_pollutant in df.columns:
                stats = df[selected_pollutant].describe()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean", f"{stats['mean']:.2f}")
                with col2:
                    st.metric("Median", f"{stats['50%']:.2f}")
                with col3:
                    st.metric("Std Dev", f"{stats['std']:.2f}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Min", f"{stats['min']:.2f}")
                with col2:
                    st.metric("25%", f"{stats['25%']:.2f}")
                with col3:
                    st.metric("75%", f"{stats['75%']:.2f}")
                
                # Distribution visualization
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(df[selected_pollutant].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_xlabel(selected_pollutant)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {selected_pollutant}')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        # Data quality info
        st.markdown("### ℹ️ Data Information")
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        
        if 'date' in df.columns:
            st.write(f"**Time Period:** {df['date'].min()} to {df['date'].max()}")
        
        if 'city' in df.columns:
            top_cities = df['city'].value_counts().head(3)
            st.write("**Top Cities:**")
            for city, count in top_cities.items():
                st.write(f"  • {city}: {count:,} records")
    
    # Divider
    st.markdown("---")
    
    # Health impact information
    st.markdown("### 🏥 Health Impact Information")
    
    if 'AQI' in df.columns:
        # Calculate AQI categories distribution
        def categorize_aqi(aqi):
            if aqi <= 50: return "Good"
            elif aqi <= 100: return "Satisfactory"
            elif aqi <= 200: return "Moderate"
            elif aqi <= 300: return "Poor"
            elif aqi <= 400: return "Very Poor"
            else: return "Severe"
        
        df['AQI_Category'] = df['AQI'].apply(categorize_aqi)
        category_counts = df['AQI_Category'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart of AQI categories
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="AQI Categories Distribution",
                color=category_counts.index,
                color_discrete_map={
                    "Good": "#00E400",
                    "Satisfactory": "#FFFF00",
                    "Moderate": "#FF7E00",
                    "Poor": "#FF0000",
                    "Very Poor": "#8F3F97",
                    "Severe": "#7E0023"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Health impact table
            st.markdown("#### Health Impact by AQI Category")
            
            health_impact_data = {
                "Good": "Minimal impact",
                "Satisfactory": "Minor breathing discomfort to sensitive people",
                "Moderate": "Breathing discomfort to people with lung disease",
                "Poor": "Breathing discomfort to most people on prolonged exposure",
                "Very Poor": "Respiratory illness on prolonged exposure",
                "Severe": "Affects healthy people and seriously impacts those with existing diseases"
            }
            
            for category, impact in health_impact_data.items():
                if category in category_counts.index:
                    count = category_counts[category]
                    percentage = (count / len(df)) * 100
                    
                    st.markdown(f"""
                    <div style="background-color: {'#e6f7e6' if category == 'Good' else '#fff7e6' if category == 'Satisfactory' else '#ffe6e6'}; 
                                padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 4px solid {'#00E400' if category == 'Good' else '#FFFF00' if category == 'Satisfactory' else '#FF0000'};">
                        <strong>{category}</strong> ({percentage:.1f}%)<br>
                        <small>{impact}</small>
                    </div>
                    """, unsafe_allow_html=True)

# ==============================================
# TAB 2: DATA EXPLORER
# ==============================================

with tab2:
    st.markdown('<h2 class="section-header">🔍 Data Explorer</h2>', unsafe_allow_html=True)
    
    # Filter controls
    st.markdown("### 🎛️ Data Filters")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        # Date range filter
        if 'date' in df.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            min_date = df['date'].min().date()
            max_date = df['date'].max().date()
            date_range = st.date_input(
                "Select Date Range",
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )
    
    with filter_col2:
        # City filter
        if 'city' in df.columns:
            cities = ['All'] + sorted(df['city'].unique().tolist())
            selected_cities = st.multiselect(
                "Select Cities",
                cities,
                default=['All']
            )
    
    with filter_col3:
        # Pollutant range filters
        pollutant = st.selectbox(
            "Select Pollutant for Range Filter",
            ['PM2_5', 'PM10', 'NO2', 'SO2', 'O3', 'CO', 'AQI']
        )
        
        if pollutant in df.columns:
            min_val = float(df[pollutant].min())
            max_val = float(df[pollutant].max())
            value_range = st.slider(
                f"{pollutant} Range",
                min_val,
                max_val,
                (min_val, max_val)
            )
    
    # Apply filters
    filtered_df = df.copy()
    
    if 'date_range' in locals() and len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= date_range[0]) & 
            (filtered_df['date'].dt.date <= date_range[1])
        ]
    
    if 'selected_cities' in locals() and 'All' not in selected_cities:
        filtered_df = filtered_df[filtered_df['city'].isin(selected_cities)]
    
    if 'value_range' in locals():
        filtered_df = filtered_df[
            (filtered_df[pollutant] >= value_range[0]) & 
            (filtered_df[pollutant] <= value_range[1])
        ]
    
    # Display filter results
    st.markdown(f"**Filtered Data:** {len(filtered_df):,} rows ({(len(filtered_df)/len(df))*100:.1f}% of total)")
    
    # Data preview with sorting
    st.markdown("### 📋 Filtered Data Preview")
    
    # Sort options
    sort_col1, sort_col2 = st.columns(2)
    with sort_col1:
        sort_by = st.selectbox(
            "Sort by",
            filtered_df.columns.tolist()
        )
    with sort_col2:
        sort_order = st.selectbox(
            "Sort order",
            ["Ascending", "Descending"]
        )
    
    # Apply sorting
    if sort_by in filtered_df.columns:
        filtered_df = filtered_df.sort_values(
            sort_by, 
            ascending=(sort_order == "Ascending")
        )
    
    # Display data with pagination
    page_size = st.slider("Rows per page", 10, 100, 20)
    
    # Calculate total pages
    total_pages = max(1, len(filtered_df) // page_size + (1 if len(filtered_df) % page_size > 0 else 0))
    
    # Page selector
    page = st.number_input("Page", 1, total_pages, 1)
    
    # Calculate slice
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(filtered_df))
    
    # Display current page
    st.dataframe(
        filtered_df.iloc[start_idx:end_idx],
        use_container_width=True,
        hide_index=True
    )
    
    # Page navigation
    st.markdown(f"**Page {page} of {total_pages}** (Rows {start_idx+1} to {end_idx} of {len(filtered_df)})")
    
    # Divider
    st.markdown("---")
    
    # Data statistics
    st.markdown("### 📊 Filtered Data Statistics")
    
    if len(filtered_df) > 0:
        # Select columns for statistics
        stat_cols = st.multiselect(
            "Select columns for statistics",
            filtered_df.select_dtypes(include=[np.number]).columns.tolist(),
            default=filtered_df.select_dtypes(include=[np.number]).columns.tolist()[:3]
        )
        
        if stat_cols:
            # Calculate statistics
            stats_df = filtered_df[stat_cols].describe().T
            stats_df['missing'] = filtered_df[stat_cols].isnull().sum()
            stats_df['missing_pct'] = (stats_df['missing'] / len(filtered_df)) * 100
            
            # Display statistics
            st.dataframe(
                stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'missing_pct']],
                use_container_width=True
            )
            
            # Download options
            st.markdown("### 💾 Download Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download filtered data
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Data (CSV)",
                    data=csv,
                    file_name=f"filtered_air_quality_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Download statistics
                stats_csv = stats_df.to_csv()
                st.download_button(
                    label="📥 Download Statistics (CSV)",
                    data=stats_csv,
                    file_name=f"air_quality_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Generate summary report
                if st.button("📄 Generate Summary Report"):
                    report = f"""
                    Air Quality Data Summary Report
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    
                    Data Overview:
                    - Total Records: {len(filtered_df):,}
                    - Date Range: {filtered_df['date'].min().date() if 'date' in filtered_df.columns else 'N/A'} to {filtered_df['date'].max().date() if 'date' in filtered_df.columns else 'N/A'}
                    - Cities: {filtered_df['city'].nunique() if 'city' in filtered_df.columns else 'N/A'}
                    
                    Statistics Summary:
                    """
                    
                    for col in stat_cols[:5]:  # Limit to first 5 columns
                        if col in filtered_df.columns:
                            report += f"\n{col}:\n"
                            report += f"  Mean: {filtered_df[col].mean():.2f}\n"
                            report += f"  Std Dev: {filtered_df[col].std():.2f}\n"
                            report += f"  Min: {filtered_df[col].min():.2f}\n"
                            report += f"  Max: {filtered_df[col].max():.2f}\n"
                    
                    st.text_area("Summary Report", report, height=300)

# ==============================================
# TAB 3: EDA & INSIGHTS
# ==============================================

with tab3:
    st.markdown('<h2 class="section-header">📈 Exploratory Data Analysis & Insights</h2>', unsafe_allow_html=True)
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Univariate Analysis", "Bivariate Analysis", "Time Series Analysis", "Correlation Analysis"]
    )
    
    if analysis_type == "Univariate Analysis":
        st.markdown("### 📊 Univariate Analysis")
        
        # Select variable for analysis
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        selected_var = st.selectbox("Select Variable", numeric_cols)
        
        if selected_var:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(
                    filtered_df, 
                    x=selected_var,
                    nbins=50,
                    title=f"Distribution of {selected_var}",
                    color_discrete_sequence=['#1f77b4']
                )
                fig.update_layout(
                    xaxis_title=selected_var,
                    yaxis_title="Frequency",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(
                    filtered_df,
                    y=selected_var,
                    title=f"Box Plot of {selected_var}",
                    color_discrete_sequence=['#2ca02c']
                )
                fig.update_layout(
                    yaxis_title=selected_var,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.markdown("#### 📈 Statistical Summary")
            stats = filtered_df[selected_var].describe()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{stats['mean']:.2f}")
            with col2:
                st.metric("Median", f"{stats['50%']:.2f}")
            with col3:
                st.metric("Std Dev", f"{stats['std']:.2f}")
            with col4:
                st.metric("IQR", f"{stats['75%'] - stats['25%']:.2f}")
    
    elif analysis_type == "Bivariate Analysis":
        st.markdown("### 🔗 Bivariate Analysis")
        
        # Select variables
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox("Select X Variable", numeric_cols)
        
        with col2:
            y_var = st.selectbox("Select Y Variable", numeric_cols)
        
        if x_var and y_var:
            # Scatter plot
            fig = px.scatter(
                filtered_df,
                x=x_var,
                y=y_var,
                title=f"{x_var} vs {y_var}",
                trendline="ols",
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate correlation
            correlation = filtered_df[[x_var, y_var]].corr().iloc[0, 1]
            st.metric("Pearson Correlation", f"{correlation:.3f}")
            
            # Interpretation
            if abs(correlation) > 0.7:
                strength = "Strong"
            elif abs(correlation) > 0.3:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            direction = "positive" if correlation > 0 else "negative"
            
            st.markdown(f"""
            <div class="info-box">
                <strong>Correlation Interpretation:</strong><br>
                {strength} {direction} correlation ({correlation:.3f})
            </div>
            """, unsafe_allow_html=True)
    
    elif analysis_type == "Time Series Analysis":
        st.markdown("### 📅 Time Series Analysis")
        
        if 'date' in filtered_df.columns:
            # Select variable for time series
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
            ts_var = st.selectbox("Select Time Series Variable", numeric_cols)
            
            if ts_var:
                # Prepare time series data
                ts_df = filtered_df.copy()
                ts_df['date'] = pd.to_datetime(ts_df['date'])
                ts_df = ts_df.set_index('date')
                
                # Resample to monthly for better visualization
                resampled = ts_df[ts_var].resample('M').mean()
                
                # Time series plot
                fig = px.line(
                    resampled.reset_index(),
                    x='date',
                    y=ts_var,
                    title=f"{ts_var} Time Series (Monthly Average)",
                    color_discrete_sequence=['#ff7f0e']
                )
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title=ts_var,
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Correlation Analysis":
        st.markdown("### 🔗 Correlation Analysis")
        
        # Select variables for correlation matrix
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = st.multiselect(
            "Select Variables for Correlation",
            numeric_cols,
            default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
        )
        
        if len(selected_cols) >= 2:
            # Calculate correlation matrix
            corr_matrix = filtered_df[selected_cols].corr()
            
            # Heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu',
                title="Correlation Matrix"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Find strongest correlations
            st.markdown("#### 🏆 Strongest Correlations")
            
            # Get upper triangle of correlation matrix
            corr_triu = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find strongest correlations
            strong_corrs = []
            for i in range(len(corr_triu.columns)):
                for j in range(i+1, len(corr_triu.columns)):
                    corr_value = corr_triu.iloc[i, j]
                    if not pd.isna(corr_value) and abs(corr_value) > 0.5:
                        strong_corrs.append({
                            'var1': corr_triu.columns[i],
                            'var2': corr_triu.columns[j],
                            'correlation': corr_value
                        })
            
            if strong_corrs:
                strong_corrs.sort(key=lambda x: abs(x['correlation']), reverse=True)
                
                for i, corr in enumerate(strong_corrs[:5]):
                    strength = "Strong" if abs(corr['correlation']) > 0.7 else "Moderate"
                    direction = "positive" if corr['correlation'] > 0 else "negative"
                    
                    st.markdown(f"""
                    <div class="info-box">
                        <strong>{corr['var1']} ↔ {corr['var2']}</strong><br>
                        Correlation: {corr['correlation']:.3f} ({strength} {direction})
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No strong correlations (|r| > 0.5) found among selected variables.")

# ==============================================
# TAB 4: MODEL INSIGHTS
# ==============================================

with tab4:
    st.markdown('<h2 class="section-header">🤖 Model Insights</h2>', unsafe_allow_html=True)
    
    # Load model configuration
    config = load_model_config()
    
    if config:
        st.markdown(f"""
        <div class="info-box">
            <strong>Model Configuration Loaded:</strong><br>
            • Target Variable: {config.get('target_column', 'N/A')}<br>
            • Features: {len(config.get('feature_names', []))}<br>
            • Best Model: {config.get('best_model', 'N/A')}<br>
            • Test Size: {config.get('test_size', 'N/A')}
        </div>
        """, unsafe_allow_html=True)
    
    # Model comparison
    st.markdown("### 🏆 Model Performance Comparison")
    
    # Load model comparison data
    try:
        comparison_paths = [
            'models/model_comparison.csv',
            'model_comparison.csv'
        ]
        
        comparison_df = None
        for path in comparison_paths:
            if os.path.exists(path):
                comparison_df = pd.read_csv(path)
                break
        
        if comparison_df is not None:
            # Display model comparison
            st.dataframe(
                comparison_df.sort_values('Test_R2', ascending=False),
                use_container_width=True,
                hide_index=True
            )
            
            # Visualize model comparison
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart of R² scores
                fig = px.bar(
                    comparison_df.sort_values('Test_R2', ascending=True),
                    y='Model',
                    x='Test_R2',
                    orientation='h',
                    title='Model Performance (Test R²)',
                    color='Test_R2',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Scatter plot of R² vs RMSE
                fig = px.scatter(
                    comparison_df,
                    x='Test_R2',
                    y='Test_RMSE',
                    size='Training_Time',
                    color='Model',
                    title='R² vs RMSE Trade-off',
                    hover_data=['Test_MAE', 'Training_Time']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Best model information
            best_model_row = comparison_df.loc[comparison_df['Test_R2'].idxmax()]
            
            st.markdown(f"""
            <div class="success-box">
                <strong>🏆 Best Performing Model:</strong> {best_model_row['Model']}<br>
                • Test R²: {best_model_row['Test_R2']:.4f}<br>
                • Test RMSE: {best_model_row['Test_RMSE']:.4f}<br>
                • Test MAE: {best_model_row['Test_MAE']:.4f}<br>
                • Training Time: {best_model_row['Training_Time']:.2f}s
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.warning("Model comparison data not found.")
    
    except Exception as e:
        st.error(f"Error loading model comparison: {str(e)}")
    
    # Feature importance
    st.markdown("### 📊 Feature Importance Analysis")
    
    # Load feature importance data
    try:
        importance_files = []
        model_dir = 'models'
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.startswith('feature_importance_') and file.endswith('.csv'):
                    importance_files.append(os.path.join(model_dir, file))
        
        if importance_files:
            # Select feature importance file
            selected_file = st.selectbox(
                "Select Feature Importance File",
                importance_files
            )
            
            if selected_file and os.path.exists(selected_file):
                importance_df = pd.read_csv(selected_file)
                
                # Display top features
                top_n = st.slider("Number of top features to show", 5, 30, 10)
                top_features = importance_df.head(top_n)
                
                # Feature importance visualization
                fig = px.bar(
                    top_features.sort_values('importance', ascending=True),
                    y='feature',
                    x='importance',
                    orientation='h',
                    title=f'Top {top_n} Feature Importances',
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance table
                st.dataframe(top_features, use_container_width=True, hide_index=True)
        
        else:
            st.info("No feature importance files found.")
    
    except Exception as e:
        st.error(f"Error loading feature importance: {str(e)}")

# ==============================================
# TAB 5: AQI PREDICTOR
# ==============================================

with tab5:
    st.markdown('<h2 class="section-header">🔮 AQI Predictor</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Use this interactive predictor to estimate Air Quality Index (AQI) based on pollutant levels.
        Adjust the sliders to simulate different pollution scenarios.
    </div>
    """, unsafe_allow_html=True)
    
    # Pollutant input controls
    st.markdown("### 📊 Pollutant Levels Input")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pm25 = st.slider(
            "PM2.5 (µg/m³)",
            min_value=0,
            max_value=500,
            value=50,
            help="Particulate matter 2.5 micrometers or smaller"
        )
        
        pm10 = st.slider(
            "PM10 (µg/m³)",
            min_value=0,
            max_value=500,
            value=100,
            help="Particulate matter 10 micrometers or smaller"
        )
    
    with col2:
        no2 = st.slider(
            "NO₂ (µg/m³)",
            min_value=0,
            max_value=200,
            value=40,
            help="Nitrogen dioxide"
        )
        
        so2 = st.slider(
            "SO₂ (µg/m³)",
            min_value=0,
            max_value=200,
            value=20,
            help="Sulfur dioxide"
        )
    
    with col3:
        o3 = st.slider(
            "O₃ (µg/m³)",
            min_value=0,
            max_value=200,
            value=50,
            help="Ozone"
        )
        
        co = st.slider(
            "CO (mg/m³)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Carbon monoxide"
        )
    
    # Additional inputs
    st.markdown("### 🏙️ Additional Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temperature = st.slider(
            "Temperature (°C)",
            min_value=0,
            max_value=50,
            value=25
        )
    
    with col2:
        humidity = st.slider(
            "Humidity (%)",
            min_value=0,
            max_value=100,
            value=60
        )
    
    with col3:
        wind_speed = st.slider(
            "Wind Speed (km/h)",
            min_value=0,
            max_value=100,
            value=10
        )
    
    # City selection
    city = st.selectbox(
        "City",
        ["Delhi", "Mumbai", "Chennai", "Kolkata", "Bangalore", "Hyderabad", "Other"]
    )
    
    # Model selection for prediction
    st.markdown("### 🤖 Prediction Model")
    
    if models:
        prediction_model = st.selectbox(
            "Select Prediction Model",
            list(models.keys())
        )
        
        model = models[prediction_model]
    else:
        st.warning("No trained models available. Using formula-based estimation.")
        model = None
    
    # Prediction button
    if st.button("🔮 Predict AQI", type="primary", use_container_width=True):
        
        if model:
            # Prepare input data for model
            try:
                # Create feature vector based on model requirements
                input_data = pd.DataFrame([{
                    'PM2_5': pm25,
                    'PM10': pm10,
                    'NO2': no2,
                    'SO2': so2,
                    'O3': o3,
                    'CO': co,
                    'temperature': temperature,
                    'humidity': humidity,
                    'wind_speed': wind_speed
                }])
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                prediction_method = "Machine Learning Model"
                
            except Exception as e:
                st.warning(f"Model prediction failed: {str(e)}. Using formula-based estimation.")
                prediction = None
                model = None
        
        if not model:
            # Formula-based AQI calculation (simplified)
            # This is a simplified version - real AQI calculation is more complex
            sub_indices = []
            
            # PM2.5 sub-index
            if pm25 <= 30:
                pm25_index = (pm25 / 30) * 50
            elif pm25 <= 60:
                pm25_index = 50 + ((pm25 - 30) / 30) * 50
            elif pm25 <= 90:
                pm25_index = 100 + ((pm25 - 60) / 30) * 100
            elif pm25 <= 120:
                pm25_index = 200 + ((pm25 - 90) / 30) * 100
            elif pm25 <= 250:
                pm25_index = 300 + ((pm25 - 120) / 130) * 100
            else:
                pm25_index = 400 + ((pm25 - 250) / 250) * 100
            
            sub_indices.append(pm25_index)
            
            # PM10 sub-index (similar logic)
            pm10_index = min(pm10 * 0.5, 500)  # Simplified
            sub_indices.append(pm10_index)
            
            # Take maximum sub-index as AQI
            prediction = max(sub_indices)
            prediction_method = "Formula-based Estimation"
        
        # Display prediction
        aqi_category = calculate_air_quality_category(prediction)
        
        st.markdown("---")
        st.markdown(f"### 📊 Prediction Results")
        
        # AQI value with color coding
        color = aqi_category['color']
        st.markdown(f"""
        <div style="background-color: {color}; color: white; padding: 2rem; border-radius: 10px; text-align: center;">
            <h1 style="margin: 0; font-size: 3rem;">{prediction:.0f}</h1>
            <h3 style="margin: 0;">{aqi_category['category']}</h3>
            <p style="margin: 1rem 0 0 0;">{prediction_method}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Health impact
        st.markdown(f"""
        <div class="info-box">
            <strong>🏥 Health Impact:</strong> {aqi_category['health_impact']}
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations based on AQI
        st.markdown("### 💡 Recommendations")
        
        if prediction <= 50:
            st.success("""
            **Good Air Quality**
            • Ideal for outdoor activities
            • Windows can be opened for ventilation
            • No restrictions needed
            """)
        elif prediction <= 100:
            st.info("""
            **Satisfactory Air Quality**
            • Generally acceptable for most people
            • Sensitive individuals should consider reducing prolonged outdoor exertion
            """)
        elif prediction <= 200:
            st.warning("""
            **Moderate Air Quality**
            • Children, elderly, and people with respiratory conditions should limit outdoor activities
            • Consider wearing masks outdoors
            • Use air purifiers indoors
            """)
        elif prediction <= 300:
            st.error("""
            **Poor Air Quality**
            • Everyone should reduce outdoor activities
            • Wear N95 masks outdoors
            • Keep windows and doors closed
            • Use air purifiers with HEPA filters
            """)
        elif prediction <= 400:
            st.error("""
            **Very Poor Air Quality**
            • Avoid all outdoor activities
            • Stay indoors with air purifiers
            • Vulnerable groups should take extra precautions
            • Consider relocating if possible
            """)
        else:
            st.error("""
            **Severe Air Quality**
            • Health emergency conditions
            • Avoid all outdoor exposure
            • Use high-efficiency air purifiers
            • Consider temporary relocation
            • Follow health authority advisories
            """)
        
        # Pollutant contribution
        st.markdown("### 📊 Pollutant Contribution")
        
        # Calculate relative contributions (simplified)
        pollutants = {
            'PM2.5': pm25,
            'PM10': pm10,
            'NO₂': no2,
            'SO₂': so2,
            'O₃': o3,
            'CO': co
        }
        
        # Normalize for visualization
        total = sum(pollutants.values())
        if total > 0:
            contributions = {k: (v / total) * 100 for k, v in pollutants.items()}
            
            fig = px.bar(
                x=list(contributions.keys()),
                y=list(contributions.values()),
                title="Relative Pollutant Contribution",
                labels={'x': 'Pollutant', 'y': 'Contribution (%)'},
                color=list(contributions.keys()),
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Download prediction report
        st.markdown("### 📄 Prediction Report")
        
        report = f"""
        AQI Prediction Report
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Input Parameters:
        • PM2.5: {pm25} µg/m³
        • PM10: {pm10} µg/m³
        • NO₂: {no2} µg/m³
        • SO₂: {so2} µg/m³
        • O₃: {o3} µg/m³
        • CO: {co} mg/m³
        • Temperature: {temperature}°C
        • Humidity: {humidity}%
        • Wind Speed: {wind_speed} km/h
        • City: {city}
        
        Prediction Results:
        • Predicted AQI: {prediction:.0f}
        • Air Quality Category: {aqi_category['category']}
        • Prediction Method: {prediction_method}
        • Health Impact: {aqi_category['health_impact']}
        
        Recommendations:
        Good (0-50): Ideal for outdoor activities
        Satisfactory (51-100): Generally acceptable
        Moderate (101-200): Sensitive groups should limit exposure
        Poor (201-300): Everyone should reduce outdoor activities
        Very Poor (301-400): Avoid outdoor activities
        Severe (401-500): Health emergency conditions
        """
        
        st.download_button(
            label="📥 Download Prediction Report",
            data=report,
            file_name=f"aqi_prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# ==============================================
# TAB 6: GEOSPATIAL VIEW
# ==============================================

with tab6:
    st.markdown('<h2 class="section-header">🗺️ Geospatial Analysis</h2>', unsafe_allow_html=True)
    
    # Check for location data
    has_location = 'latitude' in df.columns and 'longitude' in df.columns
    
    if not has_location:
        st.warning("""
        ⚠ Location data (latitude/longitude) not found in the dataset.
        
        Using simulated locations for demonstration.
        """)
        
        # Create simulated location data
        city_coordinates = {
            'Delhi': (28.7041, 77.1025),
            'Mumbai': (19.0760, 72.8777),
            'Chennai': (13.0827, 80.2707),
            'Kolkata': (22.5726, 88.3639),
            'Bangalore': (12.9716, 77.5946),
            'Hyderabad': (17.3850, 78.4867)
        }
        
        if 'city' in df.columns:
            df['latitude'] = df['city'].map(lambda x: city_coordinates.get(x, (20.0, 77.0))[0])
            df['longitude'] = df['city'].map(lambda x: city_coordinates.get(x, (20.0, 77.0))[1])
        else:
            df['latitude'] = 20.0
            df['longitude'] = 77.0
            df['city'] = 'Unknown'
    
    # Map visualization
    st.markdown("### 🗺️ Air Quality Map")
    
    # Select variable for map coloring
    map_variable = st.selectbox(
        "Select Variable for Map Coloring",
        ['AQI', 'PM2_5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']
    )
    
    if map_variable in df.columns:
        # Aggregate data by location
        if 'city' in df.columns:
            location_data = df.groupby(['city', 'latitude', 'longitude']).agg({
                map_variable: 'mean'
            }).reset_index()
        else:
            location_data = df[['latitude', 'longitude', map_variable]].copy()
            location_data['city'] = 'Unknown'
        
        # Create map
        fig = px.scatter_mapbox(
            location_data,
            lat="latitude",
            lon="longitude",
            color=map_variable,
            size=map_variable,
            hover_name="city",
            hover_data={map_variable: True},
            color_continuous_scale=px.colors.sequential.Viridis,
            zoom=4,
            height=600,
            title=f"{map_variable} Distribution Across Cities"
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":30,"l":0,"b":0}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # City comparison
        st.markdown("### 🏙️ City Comparison")
        
        if 'city' in df.columns and map_variable in df.columns:
            city_stats = df.groupby('city')[map_variable].agg(['mean', 'std', 'count']).reset_index()
            city_stats = city_stats.sort_values('mean', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart of city averages
                fig = px.bar(
                    city_stats.head(10),
                    x='city',
                    y='mean',
                    error_y='std',
                    title=f"Top 10 Cities by Average {map_variable}",
                    color='mean',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Display city statistics
                st.dataframe(
                    city_stats,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "city": "City",
                        "mean": st.column_config.NumberColumn(
                            f"Avg {map_variable}",
                            format="%.2f"
                        ),
                        "std": st.column_config.NumberColumn(
                            "Std Dev",
                            format="%.2f"
                        ),
                        "count": "Samples"
                    }
                )

# ==============================================
# FOOTER
# ==============================================

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Air Quality Analysis Dashboard</strong> | Developed for CMP7005 - Programming for Data Analysis</p>
    <p>📍 Cardiff Metropolitan University | 📅 {datetime.now().year}</p>
    <p>📧 For support or questions, contact the module leader</p>
</div>
""", unsafe_allow_html=True)

# ==============================================
# SESSION STATE MANAGEMENT
# ==============================================

if 'data' not in st.session_state:
    st.session_state.data = df
