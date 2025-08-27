#Importing the necessary libraries
import streamlit as st
import joblib
import numpy as np
import pandas as pd

#Loading your saved model. Be sure to change the path
model = joblib.load('linearANC.joblib')

# Set page title and layout
st.set_page_config(page_title="Maternal Mortality Prediction", layout="wide")

# Add a title and description
st.title('Maternal Mortality Prediction App')
st.markdown("""
This app predicts the likelihood of maternal mortality based on several factors.
Please enter the patient's information below.
""")

# Input fields for all features with improved labels and default values
st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    visits = st.number_input('Number of Visits', min_value=0, value=5, help="Number of antenatal visits.")
    age = st.number_input('Age', min_value=0, value=4, help="Patient's age.")
    education = st.number_input('Education Level', min_value=0, value=2, help="Patient's education level.")
    booking = st.number_input('Booking Status', min_value=0, value=1, help="Patient's booking status.")
    parity = st.number_input('Parity', min_value=0, value=1, help="Number of times a woman has given birth.")

with col2:
    gravida = st.number_input('Gravida', min_value=0, value=2, help="Total number of pregnancies.")
    marital = st.number_input('Marital Status', min_value=0, value=3, help="Patient's marital status.")
    mode = st.number_input('Mode of Delivery', min_value=0, value=1, help="Mode of delivery.")
    days = st.number_input('Days in Hospital', min_value=0, value=3, help="Number of days the patient stayed in the hospital.")
    complications = st.number_input('Complications', min_value=0, value=1, help="Presence of complications.")


if st.button('Predict'):
    # Prepare input features for prediction
    input_features = np.array([[visits, age, education, booking, parity, gravida, marital, mode, days, complications]])

    # Make prediction
    prediction = model.predict(input_features)

    # Display prediction with a clear message
    st.header("Prediction Result")
    if prediction[0] == 1:
        st.error("Based on the provided information, the model predicts a higher risk of maternal mortality.")
    else:
        st.success("Based on the provided information, the model predicts a lower risk of maternal mortality.")

# Add a footer
st.markdown("---")
st.markdown("Created with Streamlit")

