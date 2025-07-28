import streamlit as st
import pandas as pd
import numpy as np
import joblib
from utils.model_utils import load_model, predict_transaction

# Load data
df = pd.read_csv('data/creditcard.csv')
df_input = df.drop(columns=['Class'])
df_label = df['Class']

# Sidebar - model selector
st.sidebar.title("Model Selector")
model_choice = st.sidebar.selectbox("Choose a model", ["Random Forest", "Logistic Regression", "XGBoost"])

# Load selected model
model = load_model(model_choice)

# Calculate and display model accuracy
X = df_input
y = df_label
accuracy = model.score(X, y)
st.sidebar.markdown(f"**Accuracy:** `{accuracy:.4f}`")

# Main UI
st.title("Credit Card Fraud Detection System")
st.markdown("Detect fraud in real-time using machine learning.")

# Select transaction to predict
index = st.selectbox("Select a transaction index", df_input.index[:1000])
selected_data = df_input.loc[[index]]

st.subheader("Transaction Preview")
st.dataframe(selected_data)

# Predict button
if st.button("Detect Fraud"):
    prediction = predict_transaction(model, selected_data)
    true_label = df.loc[index, 'Class']

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("Fraud Detected")
    else:
        st.success("Legitimate Transaction")

    st.info(f"Actual Label: {'Fraud (1)' if true_label == 1 else 'Legit (0)'}")

# Footer
st.markdown("---")
st.markdown("Credit Card Fraud Detection App - By Darshan Sonara")
