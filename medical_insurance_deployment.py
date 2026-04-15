# -*- coding: utf-8 -*-

import streamlit as st
import joblib
import numpy as np

# Page settings
st.set_page_config(page_title="Medical Insurance Prediction", page_icon="💰", layout="wide")

# Load model safely
@st.cache_resource
def load_model():
    try:
        model = joblib.load("Medical_regression_model.pkl")
        return model
    except Exception as e:
        st.error("❌ Model not loaded. Check .pkl file and requirements.txt")
        st.write(e)
        return None

model = load_model()

# Title
st.markdown("<h1 style='text-align:center;color:#4CAF50;'>💰 Medical Insurance Cost Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter details to predict insurance premium</p>", unsafe_allow_html=True)

st.write("---")

# Sidebar Inputs
st.sidebar.header("Customer Details")

age = st.sidebar.number_input("Age", 18, 100, step=1)
diabetes = st.sidebar.selectbox("Diabetes", ["No", "Yes"])
bp = st.sidebar.selectbox("Blood Pressure Problems", ["No", "Yes"])
transplants = st.sidebar.selectbox("Any Transplants", ["No", "Yes"])
chronic = st.sidebar.selectbox("Chronic Diseases", ["No", "Yes"])
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200)
allergies = st.sidebar.selectbox("Known Allergies", ["No", "Yes"])
cancer = st.sidebar.selectbox("Family Cancer History", ["No", "Yes"])
surgeries = st.sidebar.number_input("Number of Major Surgeries", 0, 10, step=1)

# Encoding
diabetes = 1 if diabetes == "Yes" else 0
bp = 1 if bp == "Yes" else 0
transplants = 1 if transplants == "Yes" else 0
chronic = 1 if chronic == "Yes" else 0
allergies = 1 if allergies == "Yes" else 0
cancer = 1 if cancer == "Yes" else 0

# Summary
st.write("## Customer Summary")

col1, col2, col3 = st.columns(3)
col1.metric("Age", age)
col2.metric("Height", height)
col3.metric("Weight", weight)

st.write("---")

# Prediction
if st.button("Predict Insurance Cost", use_container_width=True):

    if model is not None:

        input_data = np.array([[age, diabetes, bp, transplants, chronic,
                                height, weight, allergies, cancer, surgeries]])

        try:
            prediction = model.predict(input_data)

            st.write("### 💰 Predicted Premium Price")
            st.success(f"Estimated Insurance Cost: ₹ {round(prediction[0], 2)}")
            st.info("Note: This is an estimated value based on given data.")

        except Exception as e:
            st.error("❌ Prediction failed")
            st.write(e)
