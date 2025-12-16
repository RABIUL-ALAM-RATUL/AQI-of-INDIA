# INDIA AIR QUALITY ANALYSIS APP
# DEVELOPED BY: MD RABIUL ALAM
# STUDENT ID: ST20316895

import streamlit as st  # Import Streamlit framework for building the web app
import pandas as pd     # Import Pandas for data manipulation and analysis
import numpy as np      # Import NumPy for numerical operations
import plotly.express as px  # Import Plotly Express for high-level interactive charts
import plotly.graph_objects as go  # Import Plotly Graph Objects for detailed custom charts
from sklearn.model_selection import train_test_split  # Import function to split data into training/testing sets
from sklearn.ensemble import RandomForestRegressor    # Import Random Forest algorithm for regression tasks
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # Import metrics to evaluate model performance
import joblib           # Import Joblib to save and load trained machine learning models
import warnings         # Import warnings library to manage system alerts

# Suppress warnings (like deprecation warnings) to keep the app interface clean
warnings.filterwarnings("ignore")

# APP CONFIGURATION

# Configure the page settings (title, layout width, and browser icon)
st.set_page_config(page_title="India Air Quality", layout="wide", page_icon="🌤️")

# Inject Custom HTML & CSS to style the app's appearance
st.markdown("""
<style>
    /* Style the main header container with a blue gradient background */
    .header {
        background: linear-gradient(90deg, #001f3f, #003366); 
        padding: 30px; /* Add padding inside the header */
        border-radius: 15px; /* Round the corners of the header */
        color: white; /* Set text color to white */
        text-align: center; /* Center-align the text */
        margin-bottom: 30px; /* Add space below the header */
        box-shadow: 0 4px 15px rgba(0,0,0,0.3); /* Add a subtle shadow effect */
    }
    
    /* Style the main title text */
    .title { font-size: 50px; font-weight: bold; margin: 0; }
    
    /* Style the subtitle text (Student ID, etc.) */
    .subtitle { font-size: 24px; margin-top: 10px; color: #e2e8f0; }
    
    /* Style the KPI cards (boxes showing numbers like Records, Cities) */
    .metric-card {
        background: linear-gradient(135deg, #0f172a, #1e293b); /* Dark gradient background */
        padding: 20px; /* Add padding inside the card */
        border-radius: 12px; /* Round the corners */
        color: white; /* Set text color to white */
        text-align: center; /* Center-align the text */
        border: 1px solid #334155; /* Add a thin border */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); /* Add a subtle shadow */
    }
    
    /* Style the number inside the KPI card */
    .metric-card h2 { margin: 10px 0 0 0; color: #38bdf8; } /* Blue color for the number */
    
    /* Style the tabs to be bold and readable */
    .stTabs [data-testid="stTab"] { font-weight: bold; font-size: 16px; }
</style>

<div class="header">
    <div class="title">India Air Quality Analysis</div>
    <div class="subtitle">CMP7005 • ST20316895 • 2025-26</div>
</div>
""", unsafe_allow_html=True)  # Render the HTML/CSS


# DATA LOADING (UPDATED FOR .GZ FILE)

# Define a function to load data and cache it (prevents reloading on every interaction)
@st.cache_data
def load_data():
    try:
        # Load the COMPRESSED .gz file using Pandas
        df = pd.read_csv("India_Air_Quality_Final_Processed.csv.gz", compression='gzip')
        
        # Clean up column names by removing leading/trailing spaces
        df.columns = [c.strip() for c in df.columns]

        # Create a dictionary to map various column name formats to a standard format
        rename_map = {'city': 'City', 'date': 'Date', 'aqi': 'AQI'}
        
        # Identify which columns need renaming (handling case sensitivity)
        new_names = {c: rename_map[c.lower()] for c in df.columns if c.lower() in rename_map}
        
        # Rename the columns in the dataframe
        df.rename(columns=new_names, inplace=True)

        # Convert 'Date' column to datetime objects if it exists
        if 'Date' in df.columns: 
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Drop rows where the 'AQI' (Target) is missing, as they are useless for training
        if 'AQI' in df.columns: 
            df = df.dropna(subset=['AQI'])
        
        # Return the cleaned dataframe
        return df
    except Exception as e:
        # If loading fails, display an error message
        st.error(f"Error loading data: {e}. Please ensure 'India_Air_Quality_Final_Processed.csv.gz' is in the directory.")
        # Return an empty dataframe to safely stop execution
        return pd.DataFrame()

# Call the load_data function to get the dataframe
df = load_data()

# Check if the dataframe is empty (load failed) and stop the app if so
if df.empty: st.stop()


# TABS & VISUALIZATION

# Create the navigation tabs for the app
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Home", "EDA", "Seasonal", "Model", "Predict", "Map", "About"])

# TAB 1: DASHBOARD
with tab1:
    st.header("Project Dashboard")  # Set header for the Dashboard tab
    
    # Calculate Total Cities (handle case if column missing)
    total_cities = df['City'].nunique() if 'City' in df.columns else 0
    # Calculate Average AQI
    avg_aqi = df['AQI'].mean() if 'AQI' in df.columns else 0
    # Calculate Maximum AQI
    max_aqi = df['AQI'].max() if 'AQI' in df.columns else 0

    # Create 4 columns for layout
    c1, c2, c3, c4 = st.columns(4)
    # Display 'Total Records' metric card
    with c1: st.markdown(f'<div class="metric-card">Records<br><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
    # Display 'Total Cities' metric card
    with c2: st.markdown(f'<div class="metric-card">Cities<br><h2>{total_cities}</h2></div>', unsafe_allow_html=True)
    # Display 'Average AQI' metric card
    with c3: st.markdown(f'<div class="metric-card">Avg AQI<br><h2>{avg_aqi:.1f}</h2></div>', unsafe_allow_html=True)
    # Display 'Max AQI' metric card
    with c4: st.markdown(f'<div class="metric-card">Max AQI<br><h2>{max_aqi:.0f}</h2></div>', unsafe_allow_html=True)

    # Check if Date and City columns exist for plotting trends
    if 'Date' in df.columns and 'City' in df.columns:
        st.markdown("### National AQI Trends")  # Add section title
        # Sample data (limit to 5000 points) to improve chart performance
        plot_data = df.sample(min(5000, len(df))) if len(df) > 5000 else df
        # Create a line chart of AQI over time
        fig = px.line(plot_data, x='Date', y='AQI', color='City', title="AQI Over Time")
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: EDA (Exploratory Data Analysis)
with tab2:
    st.header("Correlation Analysis")  # Set header
    # Select only numeric columns (excluding AQI) to find correlations between features
    numeric = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
    # Check if numeric data exists
    if not numeric.empty:
        # Create a heatmap of the correlation matrix
        # 'height=700' makes it tall enough to read easily
        fig = px.imshow(numeric.corr(), text_auto=True, color_continuous_scale='RdBu_r', height=700, aspect="auto")
        # Display the heatmap
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: SEASONAL ANALYSIS
with tab3:
    st.header("Seasonal Patterns")  # Set header
    # Check if necessary columns exist
    if 'Date' in df.columns and 'AQI' in df.columns:
        # Create a new 'Season' column by mapping month numbers to season names
        df['Season'] = df['Date'].dt.month.map({
            12:'Winter', 1:'Winter', 2:'Winter',
            3:'Spring', 4:'Spring', 5:'Spring',
            6:'Summer', 7:'Summer', 8:'Summer',
            9:'Monsoon', 10:'Monsoon', 11:'Monsoon'
        })
        # Create a box plot to show AQI distribution across seasons
        fig = px.box(df, x='Season', y='AQI', color='Season', 
                     title="Seasonal Air Quality Levels", 
                     category_orders={"Season": ["Winter", "Spring", "Summer", "Monsoon"]})
        # Display the box plot
        st.plotly_chart(fig, use_container_width=True)

# TAB 4: MODEL TRAINING (Updated with 3 Metrics)
with tab4:
    st.header("Model Training")  # Set header
    # Description text explaining the model and metrics
    st.markdown("Training Random Forest on **Scaled Data**. Showing R², MSE, and MAE.")
    
    # Create a button to start training
    if st.button("🚀 Train Model Now", type="primary"):
        # Show a spinner while training is in progress
        with st.spinner("Training Model... Please wait."):
            # Define Features (X): All numeric columns except 'AQI'
            X = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
            # Define Target (y): The 'AQI' column
            y = df['AQI']
            
            # Split data into Training (80%) and Testing (20%) sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize the Random Forest Regressor model
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            # Train the model on the training data
            model.fit(X_train, y_train)
            
            # Save the trained model to a file
            joblib.dump(model, "aqi_model.pkl")
            
            # Make predictions on the test set for evaluation
            pred = model.predict(X_test)
            # Calculate R-squared score (Accuracy)
            r2 = r2_score(y_test, pred)
            # Calculate Mean Squared Error (MSE)
            mse = mean_squared_error(y_test, pred)
            # Calculate Mean Absolute Error (MAE)
            mae = mean_absolute_error(y_test, pred)
            
            # Show success message and balloons animation
            st.success("Model Trained Successfully!")
            st.balloons()
            
            # Display the 3 metrics in separate columns
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("R² Score (Accuracy)", f"{r2:.4f}")
            with m2: st.metric("MSE (Mean Squared Error)", f"{mse:.2f}")
            with m3: st.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")

# TAB 5: PREDICT
with tab5:
    st.header("Predict Air Quality")
    try:
        # Load the pre-trained model
        model = joblib.load("aqi_model.pkl")
        
        # CRITICAL: Get the exact features the model was trained on.
        # We derive this from the dataframe used for training (X).
        X_all = df.select_dtypes(include='number').drop(columns=['AQI'], errors='ignore')
        feature_names = X_all.columns.tolist()
        
        st.info(f"ℹ️ Model expects {len(feature_names)} inputs (Z-Scores). Adjust sliders below.")
        
        # Create a dictionary to store user inputs
        user_inputs = {}
        
        # create a grid of columns for sliders
        cols = st.columns(3)
        
        # DYNAMICALLY CREATE A SLIDER FOR EVERY SINGLE FEATURE
        for i, col_name in enumerate(feature_names):
            with cols[i % 3]:
                # Default value is 0.0 (Average)
                val = st.slider(f"{col_name}", -5.0, 5.0, 0.0, 0.1)
                user_inputs[col_name] = val
        
        # Convert dictionary to a DataFrame (ensures correct column order)
        input_df = pd.DataFrame([user_inputs])
        
        if st.button("Predict AQI", type="primary", use_container_width=True):
            # Predict using the dataframe
            pred = model.predict(input_df)[0]
            
            st.divider()
            c1, c2 = st.columns([1,2])
            
            with c1: 
                # Color logic for the number
                color = "#10b981" # Green
                if pred > 100: color = "#facc15" # Yellow
                if pred > 200: color = "#f97316" # Orange
                if pred > 300: color = "#ef4444" # Red
                if pred > 400: color = "#7f1d1d" # Dark Red
                
                st.markdown(f"<h1 style='color:{color};font-size:70px;margin:0'>{pred:.0f}</h1>", unsafe_allow_html=True)
                st.markdown(f"**Predicted AQI**")
                
            with c2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", 
                    value=pred, 
                    title={'text': "Severity Level"}, 
                    gauge={
                        'axis': {'range': [0, 500]}, 
                        'bar': {'color': color}, 
                        'steps': [
                            {'range': [0,50], 'color': "#00e400"},
                            {'range': [50,100], 'color': "#ffff00"},
                            {'range': [100,200], 'color': "#ff7e00"},
                            {'range': [200,300], 'color': "#ff0000"},
                            {'range': [300,400], 'color': "#8f3f97"},
                            {'range': [400,500], 'color': "#7e0023"}
                        ]
                    }
                ))
                fig.update_layout(height=300, margin=dict(t=30,b=20,l=20,r=20))
                st.plotly_chart(fig, use_container_width=True)
                
    except FileNotFoundError:
        st.warning("⚠️ Model not found. Please go to the 'Model' tab and click 'Train Model Now'.")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
# TAB 6: MAP
with tab6:
    st.header("Pollution Hotspots")  # Set header
    # Check if City and AQI columns exist
    if 'City' in df.columns and 'AQI' in df.columns:
        # Dictionary of lat/lon coordinates for major cities
        coords = {
            'Delhi':(28.61,77.20), 'Mumbai':(19.07,72.87), 'Bengaluru':(12.97,77.59), 
            'Kolkata':(22.57, 88.36), 'Chennai':(13.08, 80.27), 'Hyderabad':(17.38, 78.48), 
            'Ahmedabad':(23.02, 72.57), 'Lucknow':(26.84, 80.94), 'Patna':(25.59, 85.13), 
            'Gurugram':(28.45, 77.02), 'Amritsar':(31.63, 74.87), 'Jaipur':(26.91, 75.78), 
            'Visakhapatnam':(17.68, 83.21), 'Thiruvananthapuram':(8.52, 76.93), 
            'Nagpur':(21.14, 79.08), 'Chandigarh':(30.73, 76.77), 'Bhopal':(23.25, 77.41), 
            'Shillong':(25.57, 91.89)
        }
        # Calculate average AQI per city
        city_stats = df.groupby('City')['AQI'].mean().reset_index()
        # Map city names to Latitude
        city_stats['lat'] = city_stats['City'].map(lambda x: coords.get(x, (None,None))[0])
        # Map city names to Longitude
        city_stats['lon'] = city_stats['City'].map(lambda x: coords.get(x, (None,None))[1])
        # Create map plot (filtering out cities without coordinates)
        fig = px.scatter_mapbox(city_stats.dropna(subset=['lat']), lat="lat", lon="lon", 
                                size="AQI", color="AQI", hover_name="City", zoom=3.5, 
                                color_continuous_scale="RdYlGn_r", title="Average AQI by City")
        # Set map style
        fig.update_layout(mapbox_style="carto-positron", height=600)
        # Display the map
        st.plotly_chart(fig, use_container_width=True)
    else: 
        # Warning if City column is missing
        st.warning("⚠️ 'City' column missing. Cannot render map.")

# TAB 7: ABOUT
with tab7:
        st.header("About This Project")
        st.markdown("""
        **CMP7005 – Programming for Data Analysis** **Student ID:** ST20316895  
        **Academic Year:** 2025–26  
        **Module Leader:** aprasad@cardiffmet.ac.uk  
        **Dataset:** India Air Quality (2015–2020)  
        **Built with:** Python • Pandas • Scikit-learn • Streamlit • Plotly  
        """)
