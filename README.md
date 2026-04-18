# 💳 Financial Fraud Detection System

## 📌 Overview
This project implements a machine learning pipeline for detecting fraudulent credit card transactions in a highly imbalanced dataset. It covers data preprocessing, imbalance handling using SMOTE, model training with Random Forest, and decision threshold tuning using precision-recall tradeoffs.

Additionally, a Streamlit application is developed to simulate real-time fraud prediction and provide an interactive interface for analyzing transaction risk.

---

## ⚠️ Problem Statement
Fraud detection is challenging because:
- Fraud cases are extremely rare (~0.17%)
- Missing fraud (false negatives) is costly
- Patterns are subtle and evolve over time

---

## 📊 Dataset
- **Source:** Credit Card Fraud Detection Dataset (public dataset)
- **Size:** ~280,000 transactions
- **Features:** V1–V28 (PCA-transformed), Time, Amount
- **Target:** Class (0 = Normal, 1 = Fraud)

---

## 🧠 Approach

### 1. Data Preprocessing
- Stratified train-test split
- Feature scaling (Amount, Time)
- Prevented data leakage (fit only on training data)

### 2. Handling Class Imbalance
- Applied **SMOTE** to balance minority class

### 3. Model
- **Random Forest Classifier**
- Handles non-linear relationships and tabular data effectively

### 4. Threshold Tuning (Key Highlight)
- Optimized decision threshold using **Precision–Recall tradeoff**
- Improved fraud detection recall significantly

---

## 📈 Results

| Metric     | Value |
|------------|------|
| Recall     | ~90% |
| Precision  | ~74% |
| ROC-AUC    | ~0.97 |

---

## 🚀 Features
- End-to-end ML pipeline
- SMOTE-based imbalance handling
- Threshold optimization for real-world performance
- Fraud probability scoring system
- Interactive **Streamlit dashboard**

---

## 🖥️ Streamlit App
- Upload transaction CSV
- Select number of rows to analyze
- View fraud predictions and probability distribution

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Streamlit
- Joblib

---

📂 Project Structure

financial-fraud-detection/
│
├── data/raw/creditcard.csv
├── notebooks/
├── src/
├── models/
├── app.py
├── test_predict.py
├── requirements.txt
└── README.md

---
# 🔮 Future Improvements

- Real-time API deployment (FastAPI)
- Model explainability (SHAP)
- Drift detection for changing fraud patterns
- Real-time streaming pipeline

# 📦 requirements.txt (clean)

```txt
pandas
numpy
scikit-learn
imbalanced-learn
streamlit
joblib
matplotlib
seaborn