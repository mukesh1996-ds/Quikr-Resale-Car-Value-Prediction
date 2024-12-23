# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 22:54:05 2024

@author: Monu Sharma
"""
import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model
model = pickle.load(open("linearegressionmodel.pkl", 'rb'))

# Load encoder categories
encoder = model.named_steps['columntransformer'].transformers_[0][1]
categories = {col: cat for col, cat in zip(['name', 'company', 'fuel_type'], encoder.categories_)}

# App title
st.title("Car Price Prediction App")

# Input fields for the user to provide car details
st.header("Enter Car Details")

name = st.text_input("Car Name (e.g., Maruti Suzuki Swift)")
company = st.text_input("Car Company (e.g., Maruti)")
year = st.number_input("Year of Manufacture", min_value=1900, max_value=2024, step=1, value=2020)
kms_driven = st.number_input("Kilometers Driven", min_value=0, step=1, value=100)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])

# Button to predict the price
if st.button("Predict Price"):
    # Validate inputs against categories
    errors = []
    if name not in categories['name']:
        errors.append(f"Car name '{name}' is not recognized. Supported names: {', '.join(categories['name'][:5])}...")
    if company not in categories['company']:
        errors.append(f"Car company '{company}' is not recognized. Supported companies: {', '.join(categories['company'][:5])}...")
    if fuel_type not in categories['fuel_type']:
        errors.append(f"Fuel type '{fuel_type}' is not recognized. Supported types: {', '.join(categories['fuel_type'])}.")
    
    # Show errors or make a prediction
    if errors:
        st.error("Error(s):\n" + "\n".join(errors))
    else:
        # Create a DataFrame with the input data
        input_data = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],
                                  columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
        try:
            # Make the prediction
            prediction = model.predict(input_data)[0]
            st.success(f"The predicted price of the car is: ₹{int(prediction):,}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Display additional information
st.sidebar.header("About")
st.sidebar.text("This app predicts the price of a used car based on its details.")
st.sidebar.text("Trained using Linear Regression.")
