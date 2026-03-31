import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import gdown

st.set_page_config(page_title="Sepsis Prediction", layout="wide")

# -------------------------------
# Load Model (SAFE + CACHED)
# -------------------------------
MODEL_PATH = "model.pkl"

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... ⏳"):
            url = "https://drive.google.com/uc?id=1xHHa-KrjDyxVWqHPtOvwKyYSA1I0LXkq"
            gdown.download(url, MODEL_PATH, quiet=True)

    return joblib.load(MODEL_PATH)

model = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("🩺 Sepsis Prediction Dashboard")
st.markdown("AI-based system to predict **sepsis risk** using patient clinical data.")

st.sidebar.title("📌 Project Info")
st.sidebar.info("""
Model: Bagging Classifier  
Metric: F1 Score Optimized  
Dataset: Clinical ICU Data  
Includes: Explainability (LIME)
""")

# -------------------------------
# Inputs
# -------------------------------
st.subheader("👤 Enter Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    Hour = st.number_input("Hour", 0, 24, 1)
    HR = st.number_input("Heart Rate", 30, 200, 80)
    O2Sat = st.number_input("Oxygen Saturation", 50, 100, 98)
    Temp = st.number_input("Temperature", 30.0, 45.0, 36.5)

with col2:
    MAP = st.number_input("MAP", 40, 150, 85)
    Resp = st.number_input("Respiration Rate", 5, 40, 18)
    BUN = st.number_input("BUN", 0.0, 50.0, 10.0)
    Chloride = st.number_input("Chloride", 80.0, 120.0, 100.0)

with col3:
    Creatinine = st.number_input("Creatinine", 0.1, 5.0, 1.0)
    Glucose = st.number_input("Glucose", 50.0, 300.0, 120.0)
    Hct = st.number_input("Hematocrit", 10.0, 60.0, 40.0)
    Hgb = st.number_input("Hemoglobin", 5.0, 20.0, 13.0)

col4, col5 = st.columns(2)

with col4:
    WBC = st.number_input("WBC", 1.0, 20.0, 7.0)
    Platelets = st.number_input("Platelets", 50.0, 500.0, 250.0)
    Age = st.number_input("Age", 0, 120, 45)

with col5:
    HospAdmTime = st.number_input("Hospital Admission Time", -100.0, 100.0, -10.0)
    ICULOS = st.number_input("ICU Length of Stay", 0, 100, 10)
    Unit = st.selectbox("Unit", [0, 1])
    Gender_1 = st.selectbox("Gender (1 = Male, 0 = Female)", [0, 1])

FEATURE_COLUMNS = [
    'Hour','HR','O2Sat','Temp','MAP','Resp','BUN','Chloride',
    'Creatinine','Glucose','Hct','Hgb','WBC','Platelets',
    'Age','HospAdmTime','ICULOS','Unit','Gender_1'
]

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict"):

    input_data = pd.DataFrame([[ 
        Hour, HR, O2Sat, Temp, MAP, Resp, BUN, Chloride,
        Creatinine, Glucose, Hct, Hgb, WBC, Platelets,
        Age, HospAdmTime, ICULOS, Unit, Gender_1
    ]], columns=FEATURE_COLUMNS)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")

    st.progress(int(probability * 100))
    st.write(f"### 🧠 Sepsis Risk Score: {probability*100:.2f}%")

    if prediction == 1:
        st.error("⚠️ High Risk of Sepsis")
    else:
        st.success("✅ No Sepsis Detected")

    st.subheader("🔍 Model Explanation")
    st.warning("SHAP not supported for Bagging model → skipped safely")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("Developed for Sepsis Prediction ML Project 🚀")
