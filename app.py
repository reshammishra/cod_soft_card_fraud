import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "fraud_model.joblib"

st.set_page_config(page_title="Fraud Detection Demo", page_icon="💳")
st.title("💳 Credit Card Fraud Detection (Educational Demo)")
st.caption("A simple Logistic Regression model for learning purposes. "
           "It is not production-ready and cannot guarantee fraud detection.")

# ---- Load model ----
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Run the Colab notebook, download "
             "fraud_model.joblib, and place it in the same folder as app.py.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model, FEATURES = bundle["pipeline"], bundle["features"]
cat_encoder = model.named_steps["preprocess"].named_transformers_["cat"]
CATEGORIES = list(cat_encoder.categories_[0])
GENDERS = list(cat_encoder.categories_[1])

# ---- Must match the notebook exactly ----
def engineer(df):
    d = pd.DataFrame(index=df.index)
    t = pd.to_datetime(df["trans_date_trans_time"])
    dob = pd.to_datetime(df["dob"])
    d["amt"] = df["amt"]
    d["city_pop"] = df["city_pop"]
    d["hour"] = t.dt.hour
    d["age"] = (t - dob).dt.days / 365.25
    lat1, lon1, lat2, lon2 = map(np.radians, [df["lat"], df["long"], df["merch_lat"], df["merch_long"]])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    d["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))
    d["category"] = df["category"]
    d["gender"] = df["gender"]
    return d[FEATURES]

threshold = st.sidebar.slider("Fraud threshold", 0.1, 0.9, 0.5, 0.05,
                              help="Flag a transaction as fraud if its fraud probability is above this value.")

tab1, tab2 = st.tabs(["Single transaction", "Upload CSV"])

# ---- Tab 1: manual entry ----
with tab1:
    c1, c2 = st.columns(2)
    amt = c1.number_input("Amount ($)", min_value=0.0, value=50.0)
    city_pop = c2.number_input("City population", min_value=0, value=20000)
    hour = c1.slider("Hour of day", 0, 23, 12)
    age = c2.number_input("Customer age", min_value=18, max_value=100, value=40)
    distance_km = c1.number_input("Distance customer to merchant (km)", min_value=0.0, value=50.0)
    category = c2.selectbox("Merchant category", CATEGORIES)
    gender = c1.selectbox("Gender", GENDERS)

    if st.button("Predict"):
        row = pd.DataFrame([{"amt": amt, "city_pop": city_pop, "hour": hour, "age": age,
                             "distance_km": distance_km, "category": category, "gender": gender}])[FEATURES]
        p = model.predict_proba(row)[0]          # [P(legit), P(fraud)]
        if p[1] >= threshold:
            st.error("⚠️ Predicted: FRAUDULENT")
        else:
            st.success("✅ Predicted: LEGITIMATE")
        st.write(f"Legitimate probability: **{p[0]:.2%}**")
        st.write(f"Fraud probability: **{p[1]:.2%}**")

# ---- Tab 2: CSV upload ----
with tab2:
    st.write("Upload a CSV with the same raw columns as `fraudTest.csv`.")
    file = st.file_uploader("Choose a CSV file", type="csv")
    if file is not None:
        data = pd.read_csv(file)
        needed = ["trans_date_trans_time", "dob", "amt", "city_pop", "category",
                  "gender", "lat", "long", "merch_lat", "merch_long"]
        missing = [c for c in needed if c not in data.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            proba = model.predict_proba(engineer(data))[:, 1]
            out = data.copy()
            out["fraud_probability"] = proba.round(4)
            out["prediction"] = np.where(proba >= threshold, "Fraud", "Legitimate")
            st.write(f"Flagged as fraud: **{(proba >= threshold).sum()}** of {len(out)} rows")
            st.dataframe(out[["amt", "category", "fraud_probability", "prediction"]].head(100))