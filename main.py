import streamlit as st
import pandas as pd
import pickle

st.title("Customer Churn Prediction")
st.subheader("By Support Vector Machine")
st.caption("Project By Nihal Tiwari")
# ===============================
# LOAD PICKLE FILES
# ===============================

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("onehot_encoder.pkl", "rb") as f:
    ohe = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# ===============================
# USER INPUT
# ===============================

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (Months)", min_value=0)

phone = st.selectbox("Phone Service", ["Yes", "No"])

multiple = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.number_input("Monthly Charges", min_value=0.0)

total = st.number_input("Total Charges", min_value=0.0)

# ===============================
# PREDICT BUTTON
# ===============================

if st.button("Predict Churn"):

    new_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone],
        "MultipleLines": [multiple],
        "InternetService": [internet],
        "OnlineSecurity": [security],
        "OnlineBackup": [backup],
        "DeviceProtection": [device],
        "TechSupport": [support],
        "StreamingTV": [tv],
        "StreamingMovies": [movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total]
    })

    # Label Encoding
    binary_cols = [
        'gender',
        'Partner',
        'Dependents',
        'PhoneService',
        'PaperlessBilling'
    ]

    for col in binary_cols:
        new_data[col] = label_encoders[col].transform(new_data[col])

    # One-Hot Encoding
    onehot_cols = [
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'Contract',
        'PaymentMethod'
    ]

    encoded = ohe.transform(new_data[onehot_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(onehot_cols)
    )

    new_data = new_data.drop(columns=onehot_cols)

    new_data = pd.concat(
        [new_data.reset_index(drop=True), encoded_df],
        axis=1
    )

    # Scaling
    numeric_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    new_data[numeric_cols] = scaler.transform(
        new_data[numeric_cols]
    )

    # Feature Order
    new_data = new_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(new_data)[0]
    probability = model.predict_proba(new_data)[0][1]

    if prediction == 1:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is not likely to churn.")

    st.write(f"**Churn Probability:** {probability*100:.2f}%")
