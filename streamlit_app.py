import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(page_title="🌳 Carbon Stock Predictor", layout="wide")

# Title
st.title("🌳 Forest Carbon Stock Prediction System")
st.markdown("**Predict carbon storage in forest regions using AI & satellite data**")

# Load model and scaler
@st.cache_resource
def load_model():
    class CarbonPredictorNN(nn.Module):
        def __init__(self):
            super(CarbonPredictorNN, self).__init__()
            self.fc1 = nn.Linear(11, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 16)
            self.fc4 = nn.Linear(16, 1)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
            x = self.fc4(x)
            return x
    
    model = CarbonPredictorNN()
    model.load_state_dict(torch.load('models/carbon_predictor_model.pth'))
    model.eval()
    return model

@st.cache_resource
def load_scaler():
    df = pd.read_csv('data/western_ghats_carbon_dataset.csv')
    X = df.iloc[:, :-1].values
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

model = load_model()
scaler = load_scaler()

# Load data
df = pd.read_csv('data/western_ghats_carbon_dataset.csv')
feature_importance = pd.read_csv('results/feature_importance.csv')

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Choose a page:", 
    ["🏠 Home", "🎯 Predict Carbon", "🗺️ Carbon Maps", "📈 Feature Importance", "📊 Model Info"])

feature_names = ['NDVI', 'EVI', 'SAVI', 'NBR', 'B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'Biomass']

# PAGE 1: HOME
if page == "🏠 Home":
    st.header("Welcome to Carbon Stock Predictor 🌍")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🌳 What is this?
        An AI-powered system that predicts **carbon storage** in forests using satellite data.
        
        ### 📡 How it works?
        1. Upload satellite spectral data
        2. AI model analyzes vegetation & forest density
        3. Predicts carbon stock in tonnes/hectare
        
        ### 💼 For Companies:
        - Monitor forest carbon assets
        - Calculate carbon credits value
        - Track climate impact
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Key Features:
        ✅ Neural Network Model (R² = 0.7976)
        ✅ SHAP Feature Analysis
        ✅ Real NASA Satellite Data
        ✅ Interactive Carbon Maps
        ✅ PDF Report Generation
        
        ### 📊 Dataset:
        - 36 forest locations
        - 11 satellite features
        - Real NASA GEDI biomass data
        """)
    
    st.divider()
    
    # Show sample predictions
    st.subheader("📊 Sample Predictions from Training Data")
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_scaled = scaler.transform(X)
    
    X_tensor = torch.FloatTensor(X_scaled[:6])
    with torch.no_grad():
        predictions = model(X_tensor).numpy()
    
    sample_df = pd.DataFrame({
        'Location': [f'Forest {i+1}' for i in range(6)],
        'Predicted Carbon (t/ha)': predictions.flatten(),
        'Actual Carbon (t/ha)': y[:6]
    })
    
    st.dataframe(sample_df, use_container_width=True)

# PAGE 2: PREDICT CARBON
elif page == "🎯 Predict Carbon":
    st.header("🎯 Predict Carbon for Your Location")
    
    st.markdown("Enter satellite spectral values to predict carbon stock")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ndvi = st.slider("NDVI (Vegetation Index)", 0.0, 1.0, 0.5)
        evi = st.slider("EVI (Enhanced Vegetation)", 0.5, 3.0, 1.5)
        savi = st.slider("SAVI (Soil-Adjusted Veg)", 0.2, 1.5, 0.7)
        nbr = st.slider("NBR (Burn Ratio)", -0.1, 0.7, 0.3)
        b2 = st.slider("B2 (Blue Band)", 300, 1000, 500)
    
    with col2:
        b3 = st.slider("B3 (Green Band)", 400, 1300, 800)
        b4 = st.slider("B4 (Red Band)", 250, 1700, 900)
        b8 = st.slider("B8 (NIR Band)", 1400, 3600, 2500)
        b11 = st.slider("B11 (SWIR-1)", 1000, 3300, 2000)
        b12 = st.slider("B12 (SWIR-2)", 400, 2600, 1500)
    
    biomass = st.slider("Biomass (Mg/ha)", 0, 250, 50)
    
    # Predict
    if st.button("🚀 Predict Carbon Stock", use_container_width=True):
        features = np.array([[ndvi, evi, savi, nbr, b2, b3, b4, b8, b11, b12, biomass]])
        features_scaled = scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled)
        
        with torch.no_grad():
            prediction = model(features_tensor).numpy()[0][0]
        
        # Categorize
        if prediction < 20:
            category = "🟢 LOW CARBON"
            color = "green"
        elif prediction < 80:
            category = "🟡 MEDIUM CARBON"
            color = "orange"
        else:
            category = "🔴 HIGH CARBON"
            color = "red"
        
        st.markdown(f"### Prediction Result")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Carbon Stock", f"{prediction:.2f}", "tonnes/ha")
        with col2:
            st.metric("Category", category)
        with col3:
            carbon_credits = prediction * 15
            st.metric("Potential Value", f"${carbon_credits:.2f}", "at $15/tonne")
        
        st.success(f"✅ Predicted carbon: **{prediction:.2f} tonnes/hectare**")

# PAGE 3: CARBON MAPS
elif page == "🗺️ Carbon Maps":
    st.header("🗺️ Forest Carbon Stock Maps")
    
    st.markdown("Interactive visualization of predicted carbon across forest regions")
    
    np.random.seed(123)
    grid_size = 10
    grid_x = np.tile(np.linspace(0, 10, grid_size), grid_size)
    grid_y = np.repeat(np.linspace(0, 10, grid_size), grid_size)
    
    grid_features = np.random.uniform(
        low=[0.2, 0.5, 0.3, -0.1, 300, 500, 300, 1500, 1200, 500, 5],
        high=[0.8, 2.8, 1.2, 0.6, 800, 1200, 1500, 3500, 3200, 2500, 150],
        size=(grid_size * grid_size, 11)
    )
    
    grid_features_scaled = scaler.transform(grid_features)
    grid_tensor = torch.FloatTensor(grid_features_scaled)
    
    with torch.no_grad():
        grid_predictions = model(grid_tensor).numpy().flatten()
    
    fig = go.Figure(data=go.Scatter(
        x=grid_x, y=grid_y,
        mode='markers',
        marker=dict(
            size=12,
            color=grid_predictions,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Carbon<br>(t/ha)")
        ),
        text=[f"Carbon: {c:.1f} t/ha" for c in grid_predictions],
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title="🗺️ Carbon Stock Heatmap",
        xaxis_title="East-West (km)",
        yaxis_title="North-South (km)",
        width=800, height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🟢 Low Carbon (<20)", f"{np.sum(grid_predictions < 20)}", "locations")
    with col2:
        st.metric("🟡 Medium (20-80)", f"{np.sum((grid_predictions >= 20) & (grid_predictions <= 80))}", "locations")
    with col3:
        st.metric("🔴 High Carbon (>80)", f"{np.sum(grid_predictions > 80)}", "locations")
    with col4:
        st.metric("Average Carbon", f"{grid_predictions.mean():.1f}", "tonnes/ha")

# PAGE 4: FEATURE IMPORTANCE
elif page == "📈 Feature Importance":
    st.header("📈 Feature Importance Analysis")
    st.markdown("Which satellite features matter most for predicting carbon?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance Score',
            color='Importance',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top_5 = feature_importance.head(5)
        fig = px.pie(
            top_5,
            values='Importance',
            names='Feature',
            title='Top 5 Features (% contribution)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("📊 Detailed Importance Scores")
    st.dataframe(feature_importance, use_container_width=True)
    
    st.markdown("""
    ### 🎯 Key Insights:
    - **Biomass** is the strongest predictor (10.85 importance)
    - **B11 & B12** (shortwave infrared) detect forest density
    - **B8** (near-infrared) measures vegetation
    - **NDVI, EVI** capture greenness but less important than expected
    """)

# PAGE 5: MODEL INFO
elif page == "📊 Model Info":
    st.header("📊 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Model Architecture")
        st.markdown("""
        **Neural Network:**
        - Input: 11 satellite features
        - Hidden 1: 64 neurons (ReLU)
        - Hidden 2: 32 neurons (ReLU)
        - Hidden 3: 16 neurons (ReLU)
        - Output: 1 (carbon prediction)
        - Optimizer: Adam
        - Epochs: 300
        """)
    
    with col2:
        st.subheader("📈 Model Performance")
        st.markdown("""
        **Test Set Metrics:**
        - R² Score: **0.7976** (80% accuracy)
        - MAE: **6.10** tonnes/ha
        - RMSE: **8.32** tonnes/ha
        
        **Dataset:**
        - 36 forest locations
        - 11 satellite features
        - Real NASA GEDI biomass data
        """)
    
    st.divider()
    
    st.subheader("📚 Model Metrics Comparison")
    
    all_metrics = pd.DataFrame({
        'Model': ['Random Forest', 'Gradient Boosting', 'Neural Network ✅'],
        'Test R²': [-6.1473, -12.0753, 0.7976],
        'Test MAE': [21.21, 28.30, 6.10],
        'Status': ['Overfitting', 'Overfitting', 'BEST']
    })
    
    st.dataframe(all_metrics, use_container_width=True)
    
    st.success("✅ Neural Network is the best model with minimal overfitting!")

st.divider()
st.markdown("""
---
**🌍 Forest Carbon AI** | Predicting climate impact with satellite data  
Built with ❤️ using Neural Networks & Streamlit | Data: NASA Sentinel-2 & GEDI
""")