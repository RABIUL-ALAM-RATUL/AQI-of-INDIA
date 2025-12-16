# INDIA AIR QUALITY ANALYSIS APP (CMP7005 WRT1)
# DEVELOPED BY: MD RABIUL ALAM
# STUDENT ID: ST20316895

# IMPORTING LIBRARIES
import streamlit as st                  # Import Streamlit: The main library to build the web app interface
import pandas as pd                     # Import Pandas: Used for reading CSVs and manipulating dataframes
import numpy as np                      # Import Numpy: Used for numerical operations like arrays and math functions
import plotly.express as px             # Import Plotly Express: High-level library for creating quick, interactive charts
import plotly.graph_objects as go       # Import Plotly Graph Objects: Lower-level library for creating detailed custom charts (like gauges)
from sklearn.model_selection import train_test_split # Import tool to split data into Training and Testing sets
from sklearn.ensemble import RandomForestRegressor   # Import the Machine Learning algorithm (Random Forest)
from sklearn.metrics import r2_score    # Import the metric to evaluate how good the model is (R-squared score)
import joblib                           # Import Joblib: Used to save the trained model to a file and load it back later
import warnings                         # Import warnings library to manage system alerts

# Suppress warnings (like deprecation warnings) to keep the app interface clean for the user
warnings.filterwarnings("ignore")

# APP CONFIGURATION & STYLING

# Configure the page settings:
# page_title: Shows in the browser tab
# layout="wide": Uses the full width of the screen (better for dashboards)
# page_icon: Sets the favicon (little icon in the browser tab)
st.set_page_config(page_title="India Air Quality", layout="wide", page_icon="🌤️")

# Inject Custom HTML & CSS to style the app and give it a professional look
# st.markdown allows us to write raw HTML/CSS directly into the app
st.markdown("""
<style>
    /* Style the main header container */
    .header {
        background: linear-gradient(90deg, #001f3f, #003366); /* Dark blue gradient background */
        padding: 35px;                                        /* Add space inside the box */
        border-radius: 18px;                                  /* Round the corners */
        color: white;                                         /* White text color */
        text-align: center;                                   /* Center align the text */
        box-shadow: 0 12px 35px rgba(0,0,0,0.4);              /* Add a shadow for 3D effect */
        margin-bottom: 40px;                                  /* Add space below the header */
    }

    /* Style the logo image inside the header */
    .header img {height: 110px; margin-right: 25px;}

    /* Style the main title text */
    .title {font-size: 56px; font-weight: bold; margin: 0;}

    /* Style the subtitle text (Module code, Student ID) */
    .subtitle {font-size: 28px; margin: 12px 0; color: #e2e8f0;}

    /* Style the Navigation Tabs */
    .stTabs [data-testid="stTab"] {
        background: #001f3f;           /* Dark blue background for tabs */
        color: white;                  /* White text */
        border-radius: 12px 12px 0 0;  /* Round only the top corners */
        padding: 16px 34px;            /* Add padding for size */
        font-weight: bold;             /* Bold text */
    }

    /* Change color of the Active (Selected) Tab */
    .stTabs [aria-selected="true"] {background: #0074D9;} /* Lighter blue for active tab */

    /* Style the Metric Cards (the boxes showing numbers) */
    .metric-card {
        background: linear-gradient(135deg, #0074D9, #001f3f); /* Diagonal gradient */
        padding: 35px;
        border-radius: 22px;
        color: white;
        text-align: center;
        box-shadow: 0 12px 30px rgba(0,0,0,0.3); /* Soft shadow */
    }
</style>

<div class="header">
    <img src="https://www.cardiffmet.ac.uk/PublishingImages/logo.png" alt="Cardiff Met">
    <div class="title">India Air Quality Analysis</div>
    <div class="subtitle">CMP7005 • ST20316895 • 2025-26</div>
</div>
""", unsafe_allow_html=True) # unsafe_allow_html=True is required to render the HTML


# DATA LOADING & CLEANING (ROBUST)

# Define a function to load data and cache it
# @st.cache_data ensures the data is loaded only once and stored in memory
# This prevents reloading the CSV every time the user clicks a button, making the app faster
@st.cache_data
def load_data():
    try:
        # Load the CSV file into a Pandas DataFrame
        df = pd.read_csv("00_MERGED_Air_Quality_India_2015_2020.csv")

        # Clean up column names by removing any leading/trailing spaces (common issue in CSVs)
        df.columns = [c.strip() for c in df.columns]

        # Robust Column Renaming
        # The code below ensures the app works even if column names have different capitalization

        # Search for a column containing 'AQI' but NOT 'BUCKET' (to avoid the categorical column)
        aqi_col = next((c for c in df.columns if 'AQI' in c.upper() and 'BUCKET' not in c.upper()), None)
        # If found, rename it to the standard 'AQI'
        if aqi_col: df.rename(columns={aqi_col: 'AQI'}, inplace=True)

        # Search for a column containing 'DATE' and rename it to 'Date'
        date_col = next((c for c in df.columns if 'DATE' in c.upper()), None)
        if date_col: df.rename(columns={date_col: 'Date'}, inplace=True)

        # Search for a column containing 'CITY' and rename it to 'City'
        city_col = next((c for c in df.columns if 'CITY' in c.upper()), None)
        if city_col: df.rename(columns={city_col: 'City'}, inplace=True)

        # Convert the 'Date' column to datetime objects so we can extract year/month later
        # errors='coerce' turns unparseable dates into NaT (Not a Time) instead of crashing
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Filtering Columns
        # We only want to keep the Date, City, AQI, and numeric columns (pollutants)
        # We select all numeric columns automatically
        keep_cols = ['Date', 'City', 'AQI'] + df.select_dtypes(include='number').columns.tolist()
        # Create a new dataframe with only these columns (using set to remove duplicates)
        df = df[list(set(keep_cols))].copy()

        # Data Cleaning
        # CRITICAL: We cannot train a model if the Target (AQI) is missing.
        # So we drop any rows where AQI is NaN.
        df = df.dropna(subset=['AQI'])

        # For feature columns (like PM2.5, NO2), we fill missing values with the Median.
        # Using Median is safer than Mean because it's not affected by outliers.
        numeric_cols = df.select_dtypes(include='number').columns.drop('AQI', errors='ignore')
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Return the cleaned dataframe
        return df
    except Exception as e:
        # If any error occurs during loading (e.g., file not found), show an error message
        st.error(f"Error loading data: {e}. Please ensure 'India_Air_Quality_Final_Processed.csv' is in the folder.")
        # Return an empty dataframe to prevent the app from crashing completely
        return pd.DataFrame()

# Call the function to load the data into the variable 'df'
df = load_data()

# Safety Check: If dataframe is empty (due to error), stop the app execution here
if df.empty:
    st.stop()


# MAIN APP TABS

# Create the navigation tabs layout
# This creates a list of tabs that the user can click to switch views
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Home", "EDA", "Seasonal", "Model", "Predict", "Map", "About"
])

# TAB 1: PROJECT DASHBOARD
with tab1:
    st.header("Project Dashboard") # Set the header for this tab

    # Create 4 columns for the metric cards
    c1, c2, c3, c4 = st.columns(4)

    # Display Key Performance Indicators (KPIs) using our custom CSS style 'metric-card'
    # .nunique() counts unique cities, .mean() calculates average, .max() finds the highest value
    with c1: st.markdown(f'<div class="metric-card">Records<br><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card">Cities<br><h2>{df["City"].nunique()}</h2></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card">Avg AQI<br><h2>{df["AQI"].mean():.1f}</h2></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card">Peak AQI<br><h2>{df["AQI"].max():.0f}</h2></div>', unsafe_allow_html=True)

    # Plot National Trends Line Chart
    # We sample 5000 points (min) to ensure the chart remains fast and responsive
    fig = px.line(df.sample(min(5000, len(df))), x='Date', y='AQI', color='City', title="National AQI Trends")
    # Display the chart using the full width of the container
    st.plotly_chart(fig, use_container_width=True)

# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
with tab2:
    st.header("Exploratory Data Analysis")

    # Select only numeric columns (excluding the target AQI) to see correlations between pollutants
    numeric = df.select_dtypes(include='number').columns.drop('AQI', errors='ignore')

    # Create a Correlation Heatmap
    # df.corr() calculates the correlation matrix
    fig = px.imshow(df[numeric].corr(), title="Pollutant Correlations", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

# TAB 3: SEASONAL ANALYSIS
with tab3:
    st.header("Seasonal Patterns")

    # Feature Engineering: Extract Month from Date and map it to a Season name
    # This helps analyze pollution based on Indian seasons (Winter, Monsoon, etc.)
    df['Season'] = df['Date'].dt.month.map({
        12:'Winter', 1:'Winter', 2:'Winter',
        3:'Spring', 4:'Spring', 5:'Spring',
        6:'Summer', 7:'Summer', 8:'Summer',
        9:'Monsoon', 10:'Monsoon', 11:'Monsoon'
    })

    # Create a Box Plot to visualize the distribution of AQI across different seasons
    # This shows the median, quartiles, and outliers for each season
    fig = px.box(df, x='Season', y='AQI', color='Season', title="AQI by Season")
    st.plotly_chart(fig, use_container_width=True)

# TAB 4: MODEL TRAINING
with tab4:
    st.header("Model Training")

    # Define the Features (X) and Target (y)
    # X: All numeric columns EXCEPT 'AQI' (the inputs)
    X = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
    # y: The 'AQI' column (what we want to predict)
    y = df['AQI']

    # Create a button to trigger the training process
    if st.button("Train Random Forest Model", type="primary"):
        with st.spinner("Training..."): # Show a loading spinner while training

            # Initialize the Random Forest Regressor
            # n_estimators=100: Create 100 decision trees
            # n_jobs=-1: Use all available CPU cores for speed
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

            # Train the model on the data
            model.fit(X, y)

            # Make predictions on the training data to calculate the score
            pred = model.predict(X)
            # Calculate R-squared score (1.0 is perfect, 0.0 is poor)
            r2 = r2_score(y, pred)

            # Save the trained model to a file so we can use it in the Predict tab
            joblib.dump(model, "aqi_model.pkl")

        # Show a success message with the R2 score
        st.success(f"Model Trained! R² = {r2:.4f}")
        # Show a celebration animation (balloons)
        st.balloons()

# TAB 5: LIVE PREDICTION
with tab5:
    st.header("Live AQI Prediction")
    try:
        # Load the trained model from the file
        model = joblib.load("aqi_model.pkl")

        # Get the feature names that the model expects (from the X dataframe)
        X_features = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
        features = X_features.columns.tolist()

        inputs = []
        # Create sliders for the top 6 features to keep the UI clean
        # Loops through the first 6 feature names
        for f in features[:6]:
            # Create a slider for each feature, default value 100
            val = st.slider(f, 0, 500, 100)
            inputs.append(val)

        # Create a button to trigger the prediction
        if st.button("Predict AQI", type="primary"):
            # Prepare the input array for the model
            # Since the model expects ALL features, we pad the remaining features with a default value of 50
            full_inputs = inputs + [50]*(len(features)-len(inputs))
            input_arr = np.array([full_inputs])

            # Use the model to predict the AQI
            pred = model.predict(input_arr)[0]

            # Display the result in a large green header
            st.markdown(f"<h1 style='color:#10b981'>Predicted AQI: {pred:.1f}</h1>", unsafe_allow_html=True)

            # Create a Gauge Chart to visualize the severity
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pred,
                title = {'text': "AQI Level"},
                gauge = {'axis': {'range': [0, 500]}, 'bar': {'color': "#10b981"}}
            ))
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        # If the model hasn't been trained yet (file not found), show an info message
        st.info(f"Please go to the 'Model' tab and train the model first. ({e})")

# TAB 6: GEOSPATIAL MAP
with tab6:
    st.header("Pollution Hotspots")

    # Define a dictionary of coordinates for major cities (Hardcoded for simplicity)
    city_coords = {
        'Delhi':(28.61,77.20), 'Mumbai':(19.07,72.87), 'Bengaluru':(12.97,77.59),
        'Kolkata':(22.57, 88.36), 'Chennai':(13.08, 80.27), 'Hyderabad':(17.38, 78.48),
        'Ahmedabad':(23.02, 72.57), 'Lucknow':(26.84, 80.94)
    }

    # Group data by City and calculate average AQI
    city_aqi = df.groupby('City')['AQI'].mean().round(0).reset_index()

    # Add Latitude and Longitude columns by mapping the city names to the dictionary
    city_aqi['lat'] = city_aqi['City'].map({k:v[0] for k,v in city_coords.items()})
    city_aqi['lon'] = city_aqi['City'].map({k:v[1] for k,v in city_coords.items()})

    # Create a Scatter Mapbox plot
    fig = px.scatter_mapbox(
        city_aqi.dropna(),       # Remove any cities without coordinates
        lat="lat", lon="lon",    # Set coordinates
        size="AQI", color="AQI", # Dot size and color depend on AQI
        hover_name="City",       # Show city name on hover
        zoom=3,                  # Initial zoom level
        title="AQI Hotspots",
        color_continuous_scale="Reds" # Red color scale for pollution
    )
    # Set the map style to a light theme
    fig.update_layout(mapbox_style="carto-positron", height=600)
    st.plotly_chart(fig, use_container_width=True)

# TAB 7: ABOUT
with tab7:
    st.header("About This Project")
    # Display project metadata using Markdown
    st.markdown("""
    **CMP7005 – Programming for Data Analysis** **Student ID:** ST20316895
    **Academic Year:** 2025–26
    **Module Leader:** aprasad@cardiffmet.ac.uk
    **Dataset:** India Air Quality (2015–2020)
    **Built with:** Python • Pandas • Scikit-learn • Streamlit • Plotly
    """)
