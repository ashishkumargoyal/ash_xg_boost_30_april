import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('ash_xgboost.joblib')

st.title("Titanic Survival Prediction App")

# User input
st.header("Enter Passenger Details:")
pclass = st.selectbox('Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.number_input('Age', min_value=0, max_value=100, value=25)
sibsp = st.number_input('No. of siblings/spouses aboard', min_value=0, max_value=10, value=0)
parch = st.number_input('No. of parents/children aboard', min_value=0, max_value=10, value=0)
fare = st.number_input('Passenger Fare', min_value=0.0, value=32.0)

# Convert categorical
sex = 0 if sex == 'male' else 1

# Predict
if st.button('Predict Survival'):
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare]
    })
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success('Passenger would Survive! 🎉')
    else:
        st.error('Passenger would not Survive. 😔')
