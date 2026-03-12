import os
import streamlit as st
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "../model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "../scaler.pkl"))

st.title("Bioinformatics Cancer Predictor")

st.write("Predict breast cancer type using biological features")

inputs = []

for i in range(30):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(value)

if st.button("Predict"):

    features = np.array(inputs).reshape(1,-1)

    prediction = model.predict(features)

    if prediction == 1:
        st.success("Prediction: Benign Tumor")
    else:
        st.error("Prediction: Malignant Tumor")
