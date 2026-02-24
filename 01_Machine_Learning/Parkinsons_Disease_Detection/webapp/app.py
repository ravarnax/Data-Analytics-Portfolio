import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ------------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Parkinson's Prediction", layout="wide")

# Paths to model files
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'models', 'model.pkl')
scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')

# Load the trained model and scaler
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("🚨 Critical Error: Model files not found. Please run 'src/train.py' first.")
    st.stop()

# ------------------------------------------------------------------------------
# 2. UI HEADER
# ------------------------------------------------------------------------------
st.title("🧠 Parkinson's Disease Prediction System")
st.markdown("""
<style>
    .big-font { font-size:20px !important; }
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">MCA Final Year Project | AI-Powered Early Diagnosis Tool</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 3. MAIN NAVIGATION (TABS)
# ------------------------------------------------------------------------------
tab1, tab2 = st.tabs(["👤 Single Patient Prediction", "📂 Batch Processing (CSV)"])

# ==============================================================================
# TAB 1: SINGLE PREDICTION
# ==============================================================================
with tab1:
    st.sidebar.header("Input Voice Parameters")
    
    def user_input_features():
        # Ideally, we should list all 22 features. 
        # For simplicity, we create sliders for key features and fill the rest with averages if needed.
        # BUT for this model to work, we need ALL 22 features in the exact order.
        
        st.sidebar.subheader("Biomedical Voice Measurements")
        
        # Group 1: Frequency (Pitch)
        mdvp_fo = st.sidebar.slider('Average Vocal Fundamental Frequency (Hz)', 88.0, 260.0, 119.0)
        mdvp_fhi = st.sidebar.slider('Max Vocal Fundamental Frequency (Hz)', 102.0, 592.0, 157.0)
        mdvp_flo = st.sidebar.slider('Min Vocal Fundamental Frequency (Hz)', 65.0, 239.0, 74.0)
        
        # Group 2: Variation
        jitter_percent = st.sidebar.slider('MDVP:Jitter(%)', 0.0, 0.04, 0.007)
        shimmer = st.sidebar.slider('MDVP:Shimmer', 0.0, 0.12, 0.04)
        nhr = st.sidebar.slider('NHR (Noise-to-Harmonic Ratio)', 0.0, 0.32, 0.02)
        hnr = st.sidebar.slider('HNR (Harmonic-to-Noise Ratio)', 8.0, 34.0, 21.0)
        
        # Group 3: Non-linear Measures (Critical for Parkinson's)
        rpde = st.sidebar.slider('RPDE (Recurrence Period Density Entropy)', 0.2, 0.7, 0.4)
        dfa = st.sidebar.slider('DFA (Detrended Fluctuation Analysis)', 0.5, 0.9, 0.8)
        spread1 = st.sidebar.slider('Spread1 (Non-linear Measure)', -8.0, -2.0, -4.8)
        spread2 = st.sidebar.slider('Spread2 (Non-linear Measure)', 0.0, 0.5, 0.2)
        d2 = st.sidebar.slider('D2 (Correlation Dimension)', 1.4, 3.7, 2.3)
        ppe = st.sidebar.slider('PPE (Pitch Period Entropy)', 0.04, 0.53, 0.28)
        
        # Note: We are using "average" dummy values for features we didn't make sliders for 
        # to prevent the UI from being 22 sliders long (which is bad UX).
        # In a real medical app, all would be input.
        
        data = {
            'MDVP:Fo(Hz)': mdvp_fo, 'MDVP:Fhi(Hz)': mdvp_fhi, 'MDVP:Flo(Hz)': mdvp_flo,
            'MDVP:Jitter(%)': jitter_percent, 'MDVP:Jitter(Abs)': 0.00004, # Fixed avg
            'MDVP:RAP': 0.003, 'MDVP:PPQ': 0.003, 'Jitter:DDP': 0.01, # Fixed avgs
            'MDVP:Shimmer': shimmer, 'MDVP:Shimmer(dB)': 0.4, # Fixed avg
            'Shimmer:APQ3': 0.02, 'Shimmer:APQ5': 0.03, 'MDVP:APQ': 0.03, 'Shimmer:DDA': 0.06,
            'NHR': nhr, 'HNR': hnr, 'RPDE': rpde, 'DFA': dfa,
            'spread1': spread1, 'spread2': spread2, 'D2': d2, 'PPE': ppe
        }
        
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    st.subheader("Patient Data Summary")
    st.write(input_df)

    if st.button('🔍 Analyze Voice Data'):
        # Scale inputs
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Diagnosis Result")
            if prediction[0] == 1:
                st.error("⚠️ POSITIVE: High Risk of Parkinson's")
                st.metric(label="Confidence Level", value=f"{prediction_proba[0][1]*100:.2f}%")
            else:
                st.success("✅ NEGATIVE: Healthy Control")
                st.metric(label="Confidence Level", value=f"{prediction_proba[0][0]*100:.2f}%")
        
        with col2:
            st.subheader("Why this prediction?")
            # Feature Importance Plot
            importance = model.feature_importances_
            feat_importances = pd.Series(importance, index=input_df.columns)
            st.bar_chart(feat_importances.nlargest(5))
            st.caption("Top 5 contributing voice features")

# ==============================================================================
# TAB 2: BATCH PREDICTION
# ==============================================================================
with tab2:
    st.header("Upload Patient Data (CSV)")
    st.write("Upload a CSV file containing voice data for multiple patients. The system will process all records instantly.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:", batch_df.head())
            
            # Remove 'name' or 'status' columns if they exist in the uploaded file
            # (We only need features for prediction)
            features_for_pred = batch_df.copy()
            if 'name' in features_for_pred.columns:
                features_for_pred = features_for_pred.drop('name', axis=1)
            if 'status' in features_for_pred.columns:
                features_for_pred = features_for_pred.drop('status', axis=1)
            
            if st.button('🚀 Process Batch Prediction'):
                # Scale
                batch_scaled = scaler.transform(features_for_pred)
                
                # Predict
                predictions = model.predict(batch_scaled)
                probabilities = model.predict_proba(batch_scaled)[:, 1] # Prob of class 1
                
                # Add results to original dataframe
                results_df = batch_df.copy()
                results_df['Prediction'] = ["Parkinson's" if p == 1 else "Healthy" for p in predictions]
                results_df['Confidence'] = [f"{prob*100:.2f}%" for prob in probabilities]
                
                st.success(f"✅ Successfully processed {len(results_df)} records!")
                st.dataframe(results_df)
                
                # Download Button
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name='parkinsons_predictions.csv',
                    mime='text/csv',
                )
                
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.warning("Please ensure your CSV matches the format of the training dataset.")

# ------------------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("MCA Final Year Project")