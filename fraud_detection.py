import streamlit as st
import pandas as pd
import joblib


# Load trained model
model = joblib.load("fraud_detection_pipeline.pkl")


# Page configuration
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🔐"
)


# Title
st.title("🔐 Fraud Detection Prediction App")

st.write(
    "Enter transaction details to predict whether "
    "the transaction is potentially fraudulent."
)

st.divider()


# Transaction type
transaction_type = st.selectbox(
    "Transaction Type",
    [
        "PAYMENT",
        "TRANSFER",
        "CASH_OUT",
        "CASH_IN",
        "DEBIT"
    ]
)


# Amount
amount = st.number_input(
    "Amount",
    min_value=0.0,
    value=1000.0
)


# Sender balances
oldbalanceOrg = st.number_input(
    "Old Balance (Sender)",
    min_value=0.0,
    value=10000.0
)

newbalanceOrig = st.number_input(
    "New Balance (Sender)",
    min_value=0.0,
    value=9000.0
)


# Receiver balances
oldbalanceDest = st.number_input(
    "Old Balance (Receiver)",
    min_value=0.0,
    value=0.0
)

newbalanceDest = st.number_input(
    "New Balance (Receiver)",
    min_value=0.0,
    value=0.0
)


# Fraud flag
isFlaggedFraud = st.selectbox(
    "Flagged Fraud",
    [0, 1]
)


# Prediction button
if st.button("Predict Fraud"):

    input_data = pd.DataFrame({
        "type": [transaction_type],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest],
        "isFlaggedFraud": [isFlaggedFraud]
    })


    # Make prediction
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]


    # Display probability
    st.subheader("Prediction Result")

    st.metric(
        "Fraud Probability",
        f"{probability * 100:.2f}%"
    )


    # Display prediction
    if prediction == 1:

        st.error(
            "⚠️ FRAUDULENT TRANSACTION DETECTED"
        )

    else:

        st.success(
            "✅ TRANSACTION APPEARS LEGITIMATE"
        )


    # Risk level
    if probability >= 0.80:

        risk = "HIGH"

    elif probability >= 0.50:

        risk = "MEDIUM"

    else:

        risk = "LOW"


    st.write(
        f"**Risk Level:** {risk}"
    )