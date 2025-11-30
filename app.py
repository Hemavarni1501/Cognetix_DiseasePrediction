import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and scaler
try:
    rf_model = joblib.load('disease_prediction_rf_model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error("Error: Model or Scaler files not found. Ensure 'disease_prediction_rf_model.joblib' and 'scaler.joblib' are in the same directory.")
    st.stop()

# Define the feature names in the correct order used during training
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# --- Streamlit UI and Logic ---
st.title('🩺 Healthcare Disease Prediction Demo')
st.subheader('Pima Indians Diabetes Dataset (Random Forest Model)')
st.markdown("""
    Enter the patient's diagnostic measurements below. 
    The model uses a Random Forest Classifier trained to predict the likelihood of diabetes.
""")

# Input fields for the 8 features
col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.slider('1. Pregnancies', 0, 17, 3)
    glucose = st.slider('2. Glucose (mg/dL)', 0, 200, 120)
    bp = st.slider('3. Blood Pressure (mmHg)', 0, 122, 70)

with col2:
    skin_thickness = st.slider('4. Skin Thickness (mm)', 0, 99, 29)
    insulin = st.slider('5. Insulin (mu U/ml)', 0, 846, 150)
    bmi = st.slider('6. BMI', 0.0, 67.1, 32.0)

with col3:
    dpf = st.slider('7. Diabetes Pedigree Function', 0.078, 2.42, 0.37, 0.01)
    age = st.slider('8. Age (Years)', 21, 81, 30)

# Bundle all inputs into a dictionary and then a DataFrame
user_data = pd.DataFrame([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]],
                         columns=feature_names)

if st.button('Predict Outcome'):
    # 1. Scaling the input data
    # The scaler must be fitted on the training data, then used to transform new data
    scaled_data = scaler.transform(user_data)
    
    # 2. Prediction
    prediction = rf_model.predict(scaled_data)
    prediction_proba = rf_model.predict_proba(scaled_data)
    
    result_proba = prediction_proba[0][1] # Probability of Class 1 (Diabetic)
    
    st.divider()
    
    # --- Display Results ---
    if prediction[0] == 1:
        st.error(f'### Prediction: HIGH RISK (Likely Diabetic) 🚨')
        st.write(f'The model predicts a **{result_proba * 100:.2f}%** probability of the patient having the disease.')
        st.info("Recommendation: Consult a medical professional for further tests.")
    else:
        st.success(f'### Prediction: LOW RISK (Non-Diabetic) ✅')
        st.write(f'The model predicts a **{(1 - result_proba) * 100:.2f}%** probability of the patient being non-diabetic.')
        st.info("The prediction is based on the input data. Continued monitoring is advised.")

# Display feature importance image at the bottom for context
st.sidebar.image('feature_importance_disease.png', caption='Model Feature Importance')