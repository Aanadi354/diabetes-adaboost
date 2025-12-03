import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and label encoder
try:
    model = joblib.load('model2.pkl')
    gender_encoder = joblib.load('label_encoder_gender.pkl')
except FileNotFoundError:
    st.error("Error: model2.pkl or label_encoder_gender.pkl not found. Make sure they are in the same directory.")
    st.stop() # Stop the app if files are not found

st.title("Diabetes Prediction App")
st.write("Enter the patient's information to get a diabetes prediction.")

# Create input fields for features
st.sidebar.header("Patient Information")

# Gender input
gender_options = ['Female', 'Male', 'Other'] # Assuming these are the classes the encoder was fitted on
gender_input = st.sidebar.selectbox("Gender", gender_options)

# Encode gender input using the loaded encoder
try:
    gender_encoded = gender_encoder.transform([gender_input])[0]
except ValueError:
    st.sidebar.warning(f"Warning: '{gender_input}' is not a recognized gender. Please select from the dropdown.")
    gender_encoded = None # Set to None to prevent prediction with invalid input

age = st.sidebar.slider("Age", 0, 100, 30)
hba1c_level = st.sidebar.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.7, step=0.1)
blood_glucose_level = st.sidebar.number_input("Blood Glucose Level", min_value=0, max_value=600, value=120)

# Make prediction when button is clicked
if st.sidebar.button("Predict"):
    if gender_encoded is not None:
        # Create a DataFrame from user inputs
        input_data = pd.DataFrame([[gender_encoded, age, hba1c_level, blood_glucose_level]],
                                  columns=['gender', 'age', 'HbA1c_level', 'blood_glucose_level'])

        # Display input data (optional, for debugging)
        # st.write("Input Data:")
        # st.dataframe(input_data)

        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.error(f"The model predicts: Diabetes (Probability: {prediction_proba[0][1]:.2f})")
            st.write("It is highly recommended to consult a medical professional for diagnosis and treatment.")
        else:
            st.success(f"The model predicts: No Diabetes (Probability: {prediction_proba[0][0]:.2f})")
            st.write("Maintain a healthy lifestyle and regular check-ups.")
    else:
        st.error("Cannot make prediction due to invalid gender input.")

st.markdown("---")
st.markdown("Disclaimer: This application is for educational and demonstrative purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.")
