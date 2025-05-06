# AI Mortgage System with Real Dataset Integration and Web Interface

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Load real dataset from GitHub or local source (for demonstration, CSV path used)
# Download from: https://github.com/Michelle-923/Malaysia-Property-Analytics
# Ensure you have a file named 'malaysia_property_data.csv' in your working directory

data = pd.read_csv("malaysia_property_data.csv")

# Inspect and preprocess dataset (example assumes columns below are available)
# Rename or engineer features based on actual dataset
kl_data = data[data['state'].str.contains("Kuala Lumpur", case=False)].dropna()
kl_data['location_score'] = kl_data['district'].astype('category').cat.codes
kl_data['property_size_sqft'] = kl_data['land_area']
kl_data['num_rooms'] = kl_data.get('num_rooms', pd.Series(np.random.randint(2, 5, size=len(kl_data))))
kl_data['estimated_value'] = kl_data['price']
kl_data['loan_amount'] = kl_data['estimated_value'] * 0.8  # Assume 80% LTV loans
kl_data['credit_score'] = np.random.randint(650, 800, size=len(kl_data))  # Simulate credit score
kl_data['annual_income'] = np.random.randint(70000, 120000, size=len(kl_data))  # Simulate income
kl_data['interest_rate'] = np.random.uniform(3.5, 4.8, size=len(kl_data))  # Simulate interest rate
kl_data['ltv'] = kl_data['loan_amount'] / kl_data['estimated_value']

# Train models
x_val = kl_data[['location_score', 'property_size_sqft', 'num_rooms']]
y_val = kl_data['estimated_value']
val_model = LinearRegression().fit(x_val, y_val)

x_rate = kl_data[['ltv', 'credit_score', 'annual_income']]
y_rate = kl_data['interest_rate']
rate_model = GradientBoostingRegressor().fit(x_rate, y_rate)

# Evaluation function
def evaluate_mortgage(location_score, size_sqft, num_rooms, loan_amount, credit_score, annual_income):
    est_value = val_model.predict([[location_score, size_sqft, num_rooms]])[0]
    ltv = loan_amount / est_value
    interest_rate = rate_model.predict([[ltv, credit_score, annual_income]])[0]
    return round(est_value, 2), round(ltv, 3), round(interest_rate, 2)

# Streamlit web interface
st.title("AI Mortgage Evaluation System - Kuala Lumpur")
st.markdown("Enter client and property details below to get mortgage recommendations.")

districts = sorted(kl_data['district'].unique())
selected_district = st.selectbox("District", districts)
location_score = kl_data[kl_data['district'] == selected_district]['location_score'].iloc[0]

size_sqft = st.number_input("Property Size (sqft)", min_value=300, max_value=5000, value=1000)
num_rooms = st.slider("Number of Rooms", 1, 10, 3)
loan_amount = st.number_input("Loan Amount (MYR)", min_value=50000, max_value=3000000, value=480000)
credit_score = st.slider("Credit Score", 300, 850, 710)
annual_income = st.number_input("Annual Income (MYR)", min_value=20000, max_value=500000, value=88000)

if st.button("Evaluate Mortgage"):
    est_value, ltv_ratio, interest_rate = evaluate_mortgage(
        location_score, size_sqft, num_rooms, loan_amount, credit_score, annual_income
    )

    st.success("Evaluation Complete")
    st.metric("Estimated Property Value", f"MYR {est_value:,.2f}")
    st.metric("Loan-to-Value (LTV) Ratio", f"{ltv_ratio:.3f}")
    st.metric("Recommended Interest Rate (%)", f"{interest_rate:.2f}")
