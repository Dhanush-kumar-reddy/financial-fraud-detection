import streamlit as st
import pandas as pd
from src.predict import predict

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("💳 Fraud Detection System")

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # 🔥 CONTROL HOW MUCH DATA TO PROCESS
    max_rows = st.slider("Select number of rows to analyze", 100, 10000, 1000)

    df_sample = df.head(max_rows)

    if st.button("🚀 Run Fraud Detection"):
        with st.spinner("Analyzing transactions..."):
            probs, preds = predict(df_sample)

        df_sample["Fraud Probability"] = probs
        df_sample["Prediction"] = preds

        # 🔥 METRICS (VERY IMPRESSIVE)
        total = len(df_sample)
        fraud_count = sum(preds)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", total)
        col2.metric("Fraud Detected", fraud_count)
        col3.metric("Fraud %", f"{(fraud_count/total)*100:.2f}%")

        st.subheader("🔍 Results Preview")
        st.dataframe(df_sample.head(20))

        st.subheader("📈 Fraud Probability Distribution")
        st.bar_chart(df_sample["Fraud Probability"])