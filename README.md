# 💳 Credit Card Fraud Detection System

A machine learning web application built with **Streamlit** and **Scikit-learn** to detect fraudulent credit card transactions in real-time.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39%2B-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6%2B-orange.svg)

---

## 📌 Project Overview

This application provides an interactive interface for predicting whether a financial transaction is legitimate or fraudulent. It uses an end-to-end scikit-learn pipeline trained on transactional features and geospatial data.

### Key Features:
- **Single Transaction Inference:** Input transaction details (amount, customer age, city population, distance, category, and gender) to get an immediate prediction and probability breakdown.
- **Batch CSV Analysis:** Upload transaction CSV batches (e.g. `sample_test_data.csv`) to classify multiple records at once with fraud count summaries and downloadable results.
- **Adjustable Threshold:** Dynamically adjust the classification probability threshold from the sidebar to tune precision vs. recall.
- **Geospatial & Behavioral Feature Engineering:** Utilizes the Haversine formula to compute distance between cardholder and merchant, along with transaction hour and demographic characteristics.

---

## 📂 Project Structure

```
credit_card_fraud/
├── app.py                  # Streamlit web application
├── fraud_model.joblib      # Pretrained Scikit-learn pipeline & features
├── sample_test_data.csv    # Sample data for testing batch upload
├── Untitled0.ipynb         # Model training & evaluation notebook
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files (large datasets, caches)
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd credit_card_fraud
```

### 2. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 🧪 Testing with Sample Data

You can use the included `sample_test_data.csv` to test the **Upload CSV** tab:
1. Open the app in your browser.
2. Navigate to the **Upload CSV** tab.
3. Upload `sample_test_data.csv`.
4. View the fraud probability scores and flagged transactions.

---

## 📊 Model Information

- **Model:** Logistic Regression with `class_weight='balanced'`
- **Preprocessing:** One-Hot Encoding for categorical attributes and Standard Scaling for numeric features
- **Engineered Features:**
  - `amt`: Transaction amount
  - `city_pop`: City population
  - `hour`: Transaction hour of day
  - `age`: Customer age at transaction time
  - `distance_km`: Haversine distance between customer location and merchant location
  - `category`: Merchant category
  - `gender`: Cardholder gender
