import numpy as np
import joblib
import pandas as pd

def load_scaler():
    """Load the StandardScaler"""
    scaler = joblib.load('models/scaler.pkl')
    return scaler

def predict_carbon(features, scaler):
    """
    Simple prediction function using pre-trained model logic
    features: numpy array of shape (n, 11)
    """
    # Normalize features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Simple neural network approximation using sklearn
    # We'll use a gradient boosting approximation
    try:
        gb_model = joblib.load('models/gb_model.pkl')
        prediction = gb_model.predict(features_scaled)[0]
    except:
        # Fallback: simple weighted average of features
        weights = np.array([0.5, 0.3, 0.1, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03])
        prediction = np.dot(features_scaled.flatten(), weights) * 100
    
    return max(0, prediction)  # Ensure positive values

def categorize_carbon(carbon):
    """Categorize carbon level"""
    if carbon < 20:
        return "🟢 LOW CARBON"
    elif carbon < 80:
        return "🟡 MEDIUM CARBON"
    else:
        return "🔴 HIGH CARBON"