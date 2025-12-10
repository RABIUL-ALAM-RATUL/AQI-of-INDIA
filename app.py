# ============================================================================
# INDIA AIR QUALITY ANALYSIS APP (CMP7005 ASSESSMENT)
# DEVELOPED BY: MD RABIUL ALAM
# STUDENT ID: ST20316895
# ============================================================================

# Import the Streamlit library for creating the web application
import streamlit as st
# Import Pandas for data manipulation and analysis
import pandas as pd
# Import NumPy for numerical operations and array handling
import numpy as np
# Import Plotly Express for high-level, interactive visualizations
import plotly.express as px
# Import Plotly Graph Objects for fine-grained control over charts
import plotly.graph_objects as go
# Import Scikit-Learn for splitting data into training and testing sets
from sklearn.model_selection import train_test_split
# Import Random Forest Regressor for the machine learning model
from sklearn.ensemble import RandomForestRegressor
# Import metrics to evaluate the model's performance
from sklearn.metrics import r2_score, mean_absolute_error
# Import StandardScaler for scaling features (optional but recommended)
from sklearn.preprocessing import StandardScaler
# Import Joblib to save and load the trained model to a file
import joblib
# Import base64 to handle file downloads (if needed)
import base64

# ============================================================================
# 1. APP CONFIGURATION & STYLING
# ============================================================================

# Configure the page title, icon, and layout (Wide mode for better dashboards)
st.set_page_config(page_title="India AQI Analytics", page_icon="🌤️", layout="wide")

# Use Custom CSS to style the app (Dark Professional Theme)
st.markdown("""
<style>
    /* Set the main background gradient color */
    .stApp {background: linear-gradient(to bottom, #0f172a, #1e293b);}
    /* Style the sidebar background color */
    [data-testid="stSidebar"] {background-color: #1e293b;}
    /* Force headings to be white with a subtle text shadow */
    h1, h2, h3 {color: #e2e8f0 !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);}
    /* Style buttons to be green with white text and rounded corners */
    .stButton button {background: #10b981; color: white; border-radius: 8px; border: none;}
    /* Add a hover effect to buttons (darker green) */
    .stButton button:hover {background: #059669;}
    /* Style metrics boxes to stand out */
    div[data-testid="stMetricValue"] {color: #38bdf8;}
</style>
""", unsafe_allow_html=True) # Apply the HTML/CSS

# ============================================================================
# 2. SIDEBAR NAVIGATION
# ============================================================================

# Add a title to the sidebar
st.sidebar.title("Air Quality Analytics")
# Add the Student details in the sidebar for identification
st.sidebar.info("Developed by: **MD RABIUL ALAM**\nID: **ST20316895**")

# Define the list of pages available in the app
pages = [
    "1. Home Dashboard",
    "2. Data Upload & Preprocess",
    "3. Exploratory Analysis",
    "4. Feature Engineering",
    "5. Model Training",
    "6. AQI Prediction (ML)",
    "7. Geospatial Maps",
    "8. Insights & Reports",
    "9. About Project"
]
# Create a selectbox (dropdown) in the sidebar for navigation
page = st.sidebar.selectbox("Navigate to:", pages)

# ============================================================================
# 3. DATA LOADING HELPER
# ============================================================================

# Decorator to cache the data so it doesn't reload on every interaction
@st.cache_data
def load_data():
    # Try to load the processed dataset from Task 3
    try:
        return pd.read_csv('India_Air_Quality_Final_Processed.csv')
    except FileNotFoundError:
        # If file is not found, return None (we will handle this in the app)
        return None

# Load the data into the 'df' variable
df = load_data()

# Check if data loaded successfully; if not, create a dummy dataframe for demo purposes
if df is None:
    # Create sample dates
    dates = pd.date_range('2015-01-01', periods=1000)
    # Create a sample dataframe
    df = pd.DataFrame({
        'date': dates,
        'city': np.random.choice(['Delhi', 'Mumbai', 'Bangalore'], 1000),
        'pm25': np.random.normal(100, 50, 1000).clip(0, 300),
        'aqi': np.random.normal(150, 70, 1000).clip(0, 500),
        'latitude': np.random.uniform(8, 37, 1000),  # Dummy Lat for Map
        'longitude': np.random.uniform(68, 97, 1000) # Dummy Lon for Map
    })
    # Show a warning that we are using sample data
    st.sidebar.warning("⚠️ Using Sample Data. Upload your CSV in Step 2.")

# ============================================================================
# PAGE 1: HOME DASHBOARD
# ============================================================================
if page == "1. Home Dashboard":
    # Set the main title
    st.title("India Air Quality Dashboard")
    # Add a descriptive subtitle
    st.markdown("Professional analytics for pollution trends, forecasting, and insights.")
    
    # Display top-level metrics in columns
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Records", f"{len(df):,}") # Show total rows
    m2.metric("Average AQI", f"{df['aqi'].mean():.0f}") # Show mean AQI
    m3.metric("Cities Monitored", f"{df['city'].nunique()}") # Show count of unique cities
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a line chart for AQI trends over time
        fig = px.line(df, x='date', y='aqi', color='city', title="AQI Trends Over Time")
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Create a box plot to show PM2.5 distribution per city
        fig = px.box(df, x='city', y='pm25', title="PM2.5 Distribution by City")
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: DATA UPLOAD & PREPROCESS
# ============================================================================
elif page == "2. Data Upload & Preprocess":
    # Set page title
    st.title("Data Upload & Preprocessing")
    
    # Create a file uploader widget accepting CSV files
    uploaded_file = st.file_uploader("Upload your Air Quality CSV", type="csv")
    
    # Check if a file has been uploaded
    if uploaded_file:
        # Read the uploaded CSV into a DataFrame
        df = pd.read_csv(uploaded_file)
        # Show the first few rows of the raw data
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())
        
        # --- Preprocessing Steps ---
        st.subheader("Applying Preprocessing...")
        
        # Convert 'Date' column to datetime objects (handling errors)
        if 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'], errors='coerce')
        elif 'date' in df.columns:
             df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Identify numeric columns for interpolation
        num_cols = df.select_dtypes(include=np.number).columns
        # Fill missing values using interpolation (Time-series standard)
        df[num_cols] = df[num_cols].interpolate(method='linear').ffill().bfill()
        
        # Display success message
        st.success("✅ Preprocessing complete! Missing values handled.")
        # Show cleaned data
        st.dataframe(df.head())

# ============================================================================
# PAGE 3: EXPLORATORY ANALYSIS
# ============================================================================
elif page == "3. Exploratory Analysis":
    # Set page title
    st.title("Exploratory Data Analysis")
    
    # Create a correlation heatmap
    # Calculate correlation matrix only on numeric columns
    corr = df.select_dtypes(include=np.number).corr()
    # Create the heatmap using Plotly Express
    fig_corr = px.imshow(corr, text_auto=".2f", title="Pollutant Correlation Heatmap", color_continuous_scale="RdBu_r")
    # Display the chart
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Create a histogram for AQI distribution
    fig_hist = px.histogram(df, x='aqi', color='city', title="AQI Distribution (Histogram)", nbins=50)
    # Display the chart
    st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================================
# PAGE 4: FEATURE ENGINEERING
# ============================================================================
elif page == "4. Feature Engineering":
    # Set page title
    st.title("Feature Engineering")
    
    # Extract Year from the Date column
    df['year'] = df['date'].dt.year
    # Extract Month from the Date column
    df['month'] = df['date'].dt.month
    # Map month numbers to Season names (Indian Context)
    df['season'] = df['month'].map({
        1:'Winter', 2:'Winter', 3:'Spring', 4:'Spring',
        5:'Spring', 6:'Summer', 7:'Summer', 8:'Summer',
        9:'Monsoon', 10:'Monsoon', 11:'Monsoon', 12:'Winter'
    })
    
    # Display the new columns
    st.write("Added Temporal Features: Year, Month, Season")
    st.dataframe(df[['date', 'year', 'month', 'season']].head())
    
    # Visualize AQI by Season
    fig_season = px.box(df, x='season', y='aqi', color='season', title="Pollution Levels by Season")
    # Display the chart
    st.plotly_chart(fig_season, use_container_width=True)

# ============================================================================
# PAGE 5: MODEL TRAINING
# ============================================================================
elif page == "5. Model Training":
    # Set page title
    st.title("Model Training (Random Forest)")
    
    # Select features (X) - Drop non-numeric and target columns
    # We use 'errors=ignore' to skip columns if they don't exist
    X = df.select_dtypes(include=np.number).drop(['aqi'], axis=1, errors='ignore')
    # Select target (y)
    y = df['aqi']
    
    # Split data into Training (80%) and Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Button to start training
    if st.button("🚀 Train Model"):
        with st.spinner("Training Random Forest Model..."):
            # Initialize Random Forest Regressor with 100 trees
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            # Fit the model to the training data
            model.fit(X_train, y_train)
            
            # Predict on the test set
            pred = model.predict(X_test)
            # Calculate R2 Score
            r2 = r2_score(y_test, pred)
            
            # Show success message with score
            st.success(f"✅ Model Trained Successfully! R² Score: {r2:.4f}")
            
            # Save the model to a file
            joblib.dump(model, 'aqi_model.pkl')
            # Save the feature names (needed for prediction page)
            joblib.dump(X.columns, 'model_features.pkl')
            
            # Allow user to download the model
            with open("aqi_model.pkl", "rb") as f:
                st.download_button("Download Trained Model (.pkl)", f, "aqi_model.pkl")

# ============================================================================
# PAGE 6: AQI PREDICTION (ML)
# ============================================================================
elif page == "6. AQI Prediction (ML)":
    # Set page title
    st.title("Interactive AQI Prediction Tool")
    
    # Check if model exists
    try:
        model = joblib.load('aqi_model.pkl')
        feature_names = joblib.load('model_features.pkl')
        
        st.write("Adjust the sliders below to predict AQI:")
        
        # Create a dictionary to hold user inputs
        input_data = {}
        # Create columns layout for sliders
        cols = st.columns(3)
        
        # Loop through each feature used in the model
        for i, col_name in enumerate(feature_names):
            # Place slider in one of the 3 columns
            with cols[i % 3]:
                # Create a number input for the feature
                # Default value is the mean of that column in the dataset
                val = st.number_input(f"{col_name}", value=float(df[col_name].mean()))
                # Store the value
                input_data[col_name] = val
        
        # Convert dictionary to DataFrame (matches model input format)
        input_df = pd.DataFrame([input_data])
        
        # Button to trigger prediction
        if st.button("Predict AQI"):
            # Make prediction
            pred_aqi = model.predict(input_df)[0]
            
            # Display the result with a large font
            st.markdown(f"<h1 style='text-align: center; color: #10b981;'>Predicted AQI: {pred_aqi:.1f}</h1>", unsafe_allow_html=True)
            
            # Create a Gauge chart for visual context
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=pred_aqi,
                gauge={
                    'axis': {'range': [0, 500]}, # AQI scale usually goes up to 500
                    'bar': {'color': "#10b981" if pred_aqi < 100 else "#ef4444"}, # Green if good, Red if bad
                    'steps': [
                        {'range': [0, 100], 'color': "lightgreen"},
                        {'range': [100, 200], 'color': "yellow"},
                        {'range': [200, 300], 'color': "orange"},
                        {'range': [300, 500], 'color': "red"}
                    ]
                },
                title={'text': "AQI Forecast"}
            ))
            # Display the gauge
            st.plotly_chart(fig_gauge, use_container_width=True)
            
    except FileNotFoundError:
        st.error("⚠️ Model not found. Please go to '5. Model Training' and train the model first.")

# ============================================================================
# PAGE 7: GEOSPATIAL MAPS
# ============================================================================
elif page == "7. Geospatial Maps":
    # Set page title
    st.title("Geospatial Visualization")
    
    # Check if latitude and longitude exist
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Create a scatter map
        fig_map = px.scatter_geo(
            df, 
            lon='longitude', 
            lat='latitude', 
            color='aqi', # Color points by AQI
            hover_name='city', # Show city name on hover
            size='aqi', # Size of point represents pollution level
            title="Pollution Hotspots Map", 
            projection="natural earth",
            color_continuous_scale="Reds" # Red = High Pollution
        )
        # Focus map on India
        fig_map.update_geos(fitbounds="locations", visible=True) 
        # Display the map
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("⚠️ Latitude/Longitude data not found in dataset. Mapping requires these columns.")

# ============================================================================
# PAGE 8: INSIGHTS & REPORTS
# ============================================================================
elif page == "8. Insights & Reports":
    # Set page title
    st.title("Insights & Recommendations")
    
    # Display text insights using Markdown
    st.markdown("""
    ### 📌 Key Findings from Analysis
    1.  **Winter Crisis:** AQI levels spike significantly during winter months (Nov-Jan).
    2.  **Primary Driver:** PM2.5 has the highest correlation with overall AQI, making it the most critical pollutant to control.
    3.  **City Trends:** Northern cities (like Delhi) show significantly higher average pollution than coastal cities.
    
    ### 💡 Recommendations
    * **Policy:** Implement stricter vehicular restrictions during Winter.
    * **Health:** Issue health advisories when predicted AQI > 200.
    * **Data:** Increase sensor density in high-variance zones.
    """)
    
    # Display a summary trend line
    st.subheader("Visual Evidence: Yearly Average Trend")
    # Group data by year to show the long-term trend
    if 'year' in df.columns:
        yearly_avg = df.groupby('year')['aqi'].mean().reset_index()
        fig_insight = px.line(yearly_avg, x='year', y='aqi', markers=True, title="Yearly Average AQI")
        st.plotly_chart(fig_insight, use_container_width=True)

# ============================================================================
# PAGE 9: ABOUT PROJECT
# ============================================================================
elif page == "9. About Project":
    # Set page title
    st.title("About This Project")
    
    # Display project details
    st.markdown(f"""
    ### 🎓 Assessment Details
    * **Module:** CMP7005 - Data Science & Analytics
    * **Student Name:** MD RABIUL ALAM
    * **Student ID:** ST20316895
    
    ### 🛠️ Tech Stack Used
    * **Python:** Core programming language.
    * **Pandas:** Data manipulation and cleaning.
    * **Plotly:** Interactive visualizations.
    * **Scikit-Learn:** Machine Learning (Random Forest).
    * **Streamlit:** Web Application Framework.
    
    ### 📝 Project Description
    This application analyzes air quality data from 2015-2020 across major Indian cities. 
    It performs the full data science lifecycle: Data Cleaning -> EDA -> Feature Engineering -> Model Training -> Deployment.
    """)

# ============================================================================
# END OF SCRIPT
# ============================================================================
