import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Activity Predictor | Simple UI",
    page_icon="🤖",
    layout="centered"
)

# --- Premium Aesthetics ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    * { font-family: 'Outfit', sans-serif; }
    
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        color: #f8fafc;
    }
    
    .main-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 40px;
        backdrop-filter: blur(20px);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    
    .prediction-title {
        font-size: 1.2rem;
        color: #94a3b8;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    
    .prediction-value {
        font-size: 3.5rem;
        font-weight: 600;
        background: linear-gradient(to right, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 24px;
    }
    
    .stNumberInput > label {
        color: #cbd5e1 !important;
        font-weight: 400 !important;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        padding: 15px;
        border-radius: 12px;
        font-weight: 600;
        letter-spacing: 0.05em;
        transition: all 0.3s ease;
        text-transform: uppercase;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# --- Constants & Asset Loading ---
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

@st.cache_resource
def load_model():
    assets = {}
    files = ['kmeans.pkl', 'scaler.pkl', 'feature_names.pkl', 'cluster_activity_mapping.pkl', 'important_features.pkl']
    try:
        for f in files:
            with open(os.path.join(MODEL_DIR, f), 'rb') as file:
                assets[f.split('.')[0]] = pickle.load(file)
        return assets
    except:
        return None

assets = load_model()

# --- Simplified Feature Mapping ---
FEATURE_MAP = {
    'tBodyAcc-mean()-X': 'Horizontal Sway',
    'tBodyAcc-mean()-Z': 'Vertical Bounce',
    'tGravityAcc-mean()-X': 'Side Leaning (Lying)',
    'tGravityAcc-mean()-Y': 'Forward/Back Tilt',
    'tBodyGyro-mean()-X': 'Rolling Speed',
    'tBodyGyro-mean()-Z': 'Spinning Speed',
    'tBodyAccMag-mean()': 'Total Energy',
    'tGravityAccMag-mean()': 'Body Weight Feel',
    'fBodyAcc-mean()-X': 'Vibration Speed',
    'fBodyAcc-std()-Y': 'Stability Score'
}

# --- Main App ---
st.title("🏃 Activity Predictor")
st.markdown("Enter simple sensor readings to identify the human activity.")

if not assets:
    st.error("Model artifacts not found. Please ensure the models directory is populated.")
    st.stop()

with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    with st.form("activity_form"):
        col1, col2 = st.columns(2)
        user_inputs = {}
        
        # We only use the 10 important features as inputs for simplicity
        # The rest will be set to 0 (neutral) for the full model prediction
        important_features = assets['important_features']
        
        for i, feat in enumerate(important_features):
            with col1 if i % 2 == 0 else col2:
                simple_name = FEATURE_MAP.get(feat, feat.replace('-', ' ').title())
                user_inputs[feat] = st.number_input(
                    simple_name, 
                    value=0.0, 
                    format="%.4f",
                    help=f"Original Technical Key: {feat}"
                )
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict Activity ✨")
        
        if submitted:
            # Prepare full feature vector (561 features)
            full_input = np.zeros((1, len(assets['feature_names'])))
            for feat, val in user_inputs.items():
                if feat in assets['feature_names']:
                    idx = assets['feature_names'].index(feat)
                    full_input[0, idx] = val
            
            # Predict
            scaled_input = assets['scaler'].transform(full_input)
            prediction_cluster = assets['kmeans'].predict(scaled_input)[0]
            activity_label = assets['cluster_activity_mapping'].get(prediction_cluster, f"Cluster {prediction_cluster}")
            
            # Success Animation
            time.sleep(0.5)
            
            st.markdown("<hr style='opacity:0.1'>", unsafe_allow_html=True)
            st.markdown(f'<div class="prediction-title">Detected Activity</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="prediction-value">{activity_label}</div>', unsafe_allow_html=True)
            
            if "WALKING" in activity_label.upper():
                st.info("💡 High frequency movement detected.")
            elif "SITTING" in activity_label.upper() or "LAYING" in activity_label.upper():
                st.info("💡 Low impact/static position detected.")
            
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><p style='text-align: center; color: #64748b; font-size: 0.8rem;'>Powered by K-Means Clustering & Scikit-Learn</p>", unsafe_allow_html=True)
