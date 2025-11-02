import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from pathlib import Path

st.title("💳 Customer Segment Predictor")

@st.cache_resource
def load_model_and_scaler():
    here = Path.cwd()
    model_path = here / "kmeans.joblib"
    scaler_path = here / "scaler.joblib" 

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()

    model = joblib.load(model_path)


    scaler = None
    if scaler_path.exists():
        try:
            s = joblib.load(scaler_path)
            if hasattr(s, "fit") and hasattr(s, "transform"):
                scaler = s
        except Exception:
            pass

    if scaler is None:
        scaler = StandardScaler()

    return model, scaler
    
segments = {
    0: "💎 Heavy Spenders",
    1: "💰 Cash-Advance Reliants", 
    2: "🎯 High-Spend Transactors",
    3: "📱 Light Users"
}


# Input fields
col1, col2 = st.columns(2)

with col1:
    balance = st.number_input("Balance", value=1000.0)
    purchases = st.number_input("Purchases", value=500.0)
    oneoff_purchases = st.number_input("One-off Purchases", value=300.0)
    installments_purchases = st.number_input("Installment Purchases", value=200.0)
    cash_advance = st.number_input("Cash Advance", value=100.0)
    credit_limit = st.number_input("Credit Limit", value=5000.0)
    payments = st.number_input("Payments", value=800.0)
    minimum_payments = st.number_input("Minimum Payments", value=200.0)
    tenure = st.selectbox("Tenure", [6, 9, 12])

with col2:
    balance_frequency = st.slider("Balance Frequency", 0.0, 1.0, 0.9)
    purchases_frequency = st.slider("Purchases Frequency", 0.0, 1.0, 0.5)
    oneoff_purchases_frequency = st.slider("One-off Frequency", 0.0, 1.0, 0.3)
    purchases_installments_frequency = st.slider("Installment Frequency", 0.0, 1.0, 0.2)
    cash_advance_frequency = st.slider("Cash Advance Frequency", 0.0, 1.0, 0.1)
    cash_advance_trx = st.number_input("Cash Advance Transactions", value=1)
    purchases_trx = st.number_input("Purchase Transactions", value=15)
    prc_full_payment = st.slider("% Full Payment", 0.0, 1.0, 0.3)

if st.button("Predict Segment"):
    # Create dataframe
    data = pd.DataFrame({
        'BALANCE': [balance],
        'BALANCE_FREQUENCY': [balance_frequency],
        'PURCHASES': [purchases],
        'ONEOFF_PURCHASES': [oneoff_purchases],
        'INSTALLMENTS_PURCHASES': [installments_purchases],
        'CASH_ADVANCE': [cash_advance],
        'PURCHASES_FREQUENCY': [purchases_frequency],
        'ONEOFF_PURCHASES_FREQUENCY': [oneoff_purchases_frequency],
        'PURCHASES_INSTALLMENTS_FREQUENCY': [purchases_installments_frequency],
        'CASH_ADVANCE_FREQUENCY': [cash_advance_frequency],
        'CASH_ADVANCE_TRX': [cash_advance_trx],
        'PURCHASES_TRX': [purchases_trx],
        'CREDIT_LIMIT': [credit_limit],
        'PAYMENTS': [payments],
        'MINIMUM_PAYMENTS': [minimum_payments],
        'PRC_FULL_PAYMENT': [prc_full_payment],
        'TENURE': [tenure]
    })
    
    
    dummy_data = pd.DataFrame({
        'BALANCE': [1000, 5000, 10000],
        'BALANCE_FREQUENCY': [0.5, 0.8, 1.0],
        'PURCHASES': [100, 1000, 5000],
        'ONEOFF_PURCHASES': [50, 500, 2000],
        'INSTALLMENTS_PURCHASES': [50, 500, 2000],
        'CASH_ADVANCE': [0, 500, 3000],
        'PURCHASES_FREQUENCY': [0.1, 0.5, 0.9],
        'ONEOFF_PURCHASES_FREQUENCY': [0.1, 0.3, 0.7],
        'PURCHASES_INSTALLMENTS_FREQUENCY': [0.1, 0.3, 0.7],
        'CASH_ADVANCE_FREQUENCY': [0.0, 0.2, 0.5],
        'CASH_ADVANCE_TRX': [0, 5, 15],
        'PURCHASES_TRX': [5, 20, 50],
        'CREDIT_LIMIT': [1000, 5000, 20000],
        'PAYMENTS': [100, 1000, 5000],
        'MINIMUM_PAYMENTS': [50, 200, 1000],
        'PRC_FULL_PAYMENT': [0.0, 0.3, 1.0],
        'TENURE': [6, 9, 12]
    })
    
    scaler.fit(dummy_data)
    scaled_data = scaler.transform(data)

    cluster = model.predict(scaled_data)[0]
    
    st.success(f"**Customer Segment: {segments[cluster]}**")
    
st.markdown("---")
