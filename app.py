import streamlit as st
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer

# ---------------- LOAD ----------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

data = load_breast_cancer()
features = data.feature_names

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Cancer Dashboard", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
.stSlider > div > div {
    color: red;
}
.block-container {
    padding-top: 1rem;
}
.metric-box {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
.result-box {
    padding: 15px;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
}
.malignant {
    background-color: #5c1f1f;
    color: #ff6b6b;
}
.benign {
    background-color: #1f5c2b;
    color: #4dff88;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🧬 Breast Cancer Prediction Dashboard")

# ---------------- INPUT UI ----------------
st.subheader("🔬 Enter Medical Parameters")

cols = st.columns(3)
input_data = []

for i, feature in enumerate(features):
    with cols[i % 3]:
        val = st.slider(feature, 0.0, 50.0, 1.0)
        input_data.append(val)

# ---------------- PREDICT ----------------
if st.button("Predict"):

    input_array = scaler.transform([input_data])
    prediction = model.predict(input_array)[0]
    prob = model.predict_proba(input_array)[0]

    st.markdown("---")

    # Result Box
    if prediction == 0:
        st.markdown(
            '<div class="result-box malignant">⚠️ Prediction: Malignant Tumor</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-box benign">✅ Prediction: Benign Tumor</div>',
            unsafe_allow_html=True
        )

    # ---------------- PROBABILITY ----------------
    st.subheader("📊 Prediction Probability")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Malignant", round(prob[0], 3))

    with col2:
        st.metric("Benign", round(prob[1], 3))