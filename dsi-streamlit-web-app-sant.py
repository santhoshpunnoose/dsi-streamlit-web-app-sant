

# Import libraries

import streamlit as st
import pandas as pd
import joblib

# Load our model pipeline object

model =joblib.load("model.joblib")

# Add Title and Instruction

st.title("Purchase Prediction Model")
st.subheader("Enter customer information and submit for likelihood to purchase")

# Age iput form
age = st.number_input(
    label = "01. Enter the customer's age",
    min_value = 18,
    max_value = 120,
    value=35)

# Gender input form 

gender = st.radio(
    label = "02. Enter the customer's gender",
    options = ['M', 'F']
    )

# Credit score input form
credit_score = st.number_input(
    label = "01. Enter the customer's credit score",
    min_value = 0,
    max_value = 1000,
    value=500
    )

# Submit inputs to model

if st.button("Submit For Prediction"):
    
    # Store our data in dataframe for prediction
    new_data = pd.DataFrame({"age": [age], "gender": [gender], "credit_score": [credit_score]})
    
    
    # Apply model pipeline to the input data and extract probability prediction
    
    pred_proba = model.predict_proba(new_data)[0][1]
    
    # Output prediction
    
    st.subheader(f"Based on these customer attribute our model predicts a purchase probability of {pred_proba:.0%}")
    
    
    

