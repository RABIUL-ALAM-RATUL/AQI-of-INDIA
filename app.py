import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ============================================================================
# 1. APP CONFIGURATION & SETUP
# ============================================================================
st.set_page_config(
    page_title="Air Quality Forecasting App",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main {background-color: #f8fafc;}
    h1, h2, h3 {color: #1e293b; font-family: 'Arial', sans-serif;}
    .stMetric {background-color: #ffffff; padding: 15px; border-radius: 5px; border-left: 5px solid #3b82f6; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# 2. DATA LOADING (CACHED)
# ============================================================================
@st.cache_data
def load_data():
    # Load the processed "Golden Dataset"
    # Ensure this file is in the same directory
    df = pd.read_csv('India_Air_Quality_Final_Processed.csv')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ Data file not found! Please ensure 'India_Air_Quality_Final_Processed.csv' is in the directory.")
    st.stop()

# ============================================================================
# 3. SIDEBAR NAVIGATION [cite: 58]
# ============================================================================
st.sidebar.title("Navigation")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3222/3222800.png", width=100) # Placeholder icon
page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Data Analysis", "Model & Prediction"])

st.sidebar.info("Developed for CMP7005 Assessment [cite: 8]")

# ============================================================================
# PAGE 1: DATA OVERVIEW 
# ============================================================================
if page == "Data Overview":
    st.title("📊 Data Overview")
    st.markdown("### India Air Quality Dataset (2015-2020)")
    st.write("This section provides a summary of the processed dataset used for analysis and modeling.")

    # Top Level Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{df.shape[0]:,}")
    col2.metric("Total Columns", f"{df.shape[1]}")
    col3.metric("Cities Covered", "12") # Update based on your actual data
    col4.metric("Pollutants Tracked", "12")

    # Data Sample
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Statistical Summary
    st.subheader("📈 Statistical Summary")
    st.write(df.describe())

# ============================================================================
# PAGE 2: EXPLORATORY DATA ANALYSIS (EDA) 
# ============================================================================
elif page == "Exploratory Data Analysis":
    st.title("🔍 Exploratory Data Analysis")
    st.write("Visualizing trends, correlations, and seasonal patterns in air quality.")

    # 1. Distribution of AQI
    st.subheader("1. Distribution of AQI (Target Variable)")
    fig_dist = px.histogram(df, x="aqi", nbins=50, title="AQI Distribution", color_discrete_sequence=['#3b82f6'])
    st.plotly_chart(fig_dist, use_container_width=True)

    # 2. Correlation Heatmap
    st.subheader("2. Pollutant Correlations")
    # Calculate correlation only on numeric columns
    corr_matrix = df.select_dtypes(include=np.number).corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

    # 3. Seasonal Trends (Interactive)
    st.subheader("3. Seasonal Analysis")
    if 'season' in df.columns:
        fig_season = px.box(df, x="season", y="aqi", color="season", title="AQI Levels by Season")
        st.plotly_chart(fig_season, use_container_width=True)
    else:
        st.warning("Season column not found. Ensure feature engineering is complete.")

    # 4. 3D PCA (Your Advanced Visual)
    st.subheader("4. 3D PCA: Data Structure Analysis")
    # For performance, we might sample the data or use pre-calculated components if available
    # Here we perform a quick on-the-fly PCA for visualization
    from sklearn.decomposition import PCA
    features = ['pm25','pm10','no','no2','nox','nh3','co','so2','o3','benzene','toluene','xylene']
    features = [f for f in features if f in df.columns]
    
    if len(features) > 0:
        X_pca = df[features].fillna(0) # Safety fill
        pca = PCA(n_components=3)
        components = pca.fit_transform(X_pca.sample(1000)) # Sample 1000 rows for speed
        df_pca = pd.DataFrame(components, columns=['PC1', 'PC2', 'PC3'])
        
        fig_3d = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', opacity=0.6, title="3D Projection of Pollution Data (Sampled)")
        st.plotly_chart(fig_3d, use_container_width=True)

# ============================================================================
# PAGE 3: MODELLING & PREDICTION 
# ============================================================================
elif page == "Model & Prediction":
    st.title("🤖 Model Building & Prediction")
    st.write("Train machine learning models and predict AQI based on pollutant levels.")

    # 1. Model Setup
    st.sidebar.header("Model Configuration")
    split_size = st.sidebar.slider("Train/Test Split Ratio", 0.1, 0.4, 0.2)
    model_choice = st.sidebar.selectbox("Choose Model", ["Linear Regression", "Random Forest"])

    # Prepare Data
    target = 'aqi'
    features_list = ['pm25','pm10','no','no2','nox','nh3','co','so2','o3','benzene','toluene','xylene']
    features_list = [f for f in features_list if f in df.columns]
    
    X = df[features_list]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)

    # 2. Train Model Button
    if st.button("🚀 Train Model"):
        with st.spinner("Training model... please wait..."):
            if model_choice == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            st.success(f"{model_choice} Trained Successfully!")
            
            # Display Metrics
            col1, col2 = st.columns(2)
            col1.metric("R² Score (Accuracy)", f"{r2:.4f}")
            col2.metric("RMSE (Error)", f"{rmse:.2f}")

            # Save model to session state for prediction usage
            st.session_state['model'] = model
            st.session_state['features'] = features_list

            # Actual vs Predicted Plot
            fig_res = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual AQI', 'y': 'Predicted AQI'}, title="Actual vs Predicted")
            fig_res.add_shape(type="line", line=dict(dash='dash'), x0=y.min(), y0=y.max(), x1=y.min(), y1=y.max())
            st.plotly_chart(fig_res, use_container_width=True)

    # 3. Interactive Prediction Interface
    st.markdown("---")
    st.subheader("🔮 Live Prediction")
    
    if 'model' in st.session_state:
        st.write("Adjust the sliders below to simulate pollutant levels and predict AQI.")
        
        # Dynamic Input Widgets
        input_data = {}
        cols = st.columns(3)
        for i, feature in enumerate(st.session_state['features']):
            with cols[i % 3]:
                val = st.number_input(f"{feature}", value=float(df[feature].mean()))
                input_data[feature] = val
        
        if st.button("Predict AQI"):
            input_df = pd.DataFrame([input_data])
            prediction = st.session_state['model'].predict(input_df)[0]
            
            # Display Result with Color Coding
            color = "green" if prediction < 100 else "orange" if prediction < 200 else "red"
            st.markdown(f"### Predicted AQI: <span style='color:{color}'>{prediction:.2f}</span>", unsafe_allow_html=True)
    else:
        st.info("Please train a model above to unlock the prediction feature.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("© 2025 Cardiff Metropolitan University | CMP7005 Assessment [cite: 4]")
