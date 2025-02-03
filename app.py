import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Get the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Model and Preprocessor
model_path = os.path.join(BASE_DIR, "model", "regression_model.joblib")
preprocessor_path = os.path.join(BASE_DIR, "preprocessor.joblib")

if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
    st.error("Model or preprocessor not found! Please train the model first.")
    st.stop()

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

st.title("Sales Prediction App")

# Input Form
feature_inputs = {}
st.write("Enter the values for prediction:")

# Dynamically generate input fields based on dataset columns
feature_names = ["Feature1", "Feature2", "Feature3"]  # Replace with actual feature names
for feature in feature_names:
    feature_inputs[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Predict
if st.button("Predict Sales"):
    input_df = pd.DataFrame([feature_inputs])
    input_processed = preprocessor.transform(input_df)
    prediction = model.predict(input_processed)[0]

    st.write(f"Predicted Sales: {prediction:.2f} units")