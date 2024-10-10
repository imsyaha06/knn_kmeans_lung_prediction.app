######################################  BOTH MODELS SIDE BY SIDE ################################3

import streamlit as st
import numpy as np
import joblib

# Load the trained KMeans and KNN models, scaler, and feature names
kmeans_model = joblib.load('kmeans_lung_cancer_model.pkl')
knn_model = joblib.load('knn_lung_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

with open('features.txt', 'r') as f:
    feature_names = f.read().splitlines()

# Define a function to make predictions using KMeans
def predict_with_kmeans(input_data):
    input_data_scaled = scaler.transform(np.array(input_data).reshape(1, -1))
    cluster = kmeans_model.predict(input_data_scaled)
    lung_cancer_result = "YES" if cluster[0] == 0 else "NO"
    return lung_cancer_result

# Define the Streamlit app
def main():
    st.title("Lung Cancer Prediction App          (KMeans vs KNN)")

    st.markdown("""
    This app allows you to make predictions about lung cancer using two different models: 
    **KMeans** and **KNN**. You can input the same data for both models and compare their predictions side by side.
    """)

    # Create two columns for side-by-side display of the KMeans and KNN models
    col1, col2 = st.columns(2)

    # Input fields in both columns
    with col1:
        st.header("KMeans Model")
        st.write("Input values for KMeans model:")

        # Collect user inputs
        gender_kmeans = st.selectbox('Gender', [1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female', key='gender_kmeans')
        age_kmeans = st.slider('Age', min_value=21, max_value=100, value=30, key='age_kmeans')
        smoking_kmeans = st.selectbox('Smoking', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='smoking_kmeans')
        yellow_fingers_kmeans = st.selectbox('Yellow Fingers', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='yellow_kmeans')
        anxiety_kmeans = st.selectbox('Anxiety', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='anxiety_kmeans')
        peer_pressure_kmeans = st.selectbox('Peer Pressure', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='peer_kmeans')
        chronic_disease_kmeans = st.selectbox('Chronic Disease', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='chronic_kmeans')
        fatigue_kmeans = st.selectbox('Fatigue', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='fatigue_kmeans')
        allergy_kmeans = st.selectbox('Allergy', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='allergy_kmeans')
        wheezing_kmeans = st.selectbox('Wheezing', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='wheezing_kmeans')
        alcohol_kmeans = st.selectbox('Alcohol Consuming', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='alcohol_kmeans')
        coughing_kmeans = st.selectbox('Coughing', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='coughing_kmeans')
        breath_kmeans = st.selectbox('Shortness of Breath', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='breath_kmeans')
        swallowing_kmeans = st.selectbox('Swallowing Difficulty', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='swallowing_kmeans')
        chest_pain_kmeans = st.selectbox('Chest Pain', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='chest_kmeans')

        if st.button("Predict with KMeans", key='predict_kmeans'):
            input_data_kmeans = [
                gender_kmeans, age_kmeans, smoking_kmeans, yellow_fingers_kmeans, anxiety_kmeans, 
                peer_pressure_kmeans, chronic_disease_kmeans, fatigue_kmeans, allergy_kmeans, 
                wheezing_kmeans, alcohol_kmeans, coughing_kmeans, breath_kmeans, 
                swallowing_kmeans, chest_pain_kmeans
            ]
            result_kmeans = predict_with_kmeans(input_data_kmeans)
            st.success(f"KMeans Model predicts: Lung Cancer: {result_kmeans}")

    with col2:
        st.header("KNN Model")
        st.write("Input values for KNN model:")

        # Collect user inputs (similar inputs to the left column for consistency)
        gender_knn = st.selectbox('Gender', [1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female', key='gender_knn')
        age_knn = st.slider('Age', min_value=21, max_value=100, value=30, key='age_knn')
        smoking_knn = st.selectbox('Smoking', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='smoking_knn')
        yellow_fingers_knn = st.selectbox('Yellow Fingers', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='yellow_knn')
        anxiety_knn = st.selectbox('Anxiety', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='anxiety_knn')
        peer_pressure_knn = st.selectbox('Peer Pressure', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='peer_knn')
        chronic_disease_knn = st.selectbox('Chronic Disease', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='chronic_knn')
        fatigue_knn = st.selectbox('Fatigue', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='fatigue_knn')
        allergy_knn = st.selectbox('Allergy', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='allergy_knn')
        wheezing_knn = st.selectbox('Wheezing', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='wheezing_knn')
        alcohol_knn = st.selectbox('Alcohol Consuming', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='alcohol_knn')
        coughing_knn = st.selectbox('Coughing', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='coughing_knn')
        breath_knn = st.selectbox('Shortness of Breath', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='breath_knn')
        swallowing_knn = st.selectbox('Swallowing Difficulty', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='swallowing_knn')
        chest_pain_knn = st.selectbox('Chest Pain', [1, 2], format_func=lambda x: 'Yes' if x == 1 else 'No', key='chest_knn')

        if st.button("Predict with KNN", key='predict_knn'):
            input_data_knn = [
                gender_knn, age_knn, smoking_knn, yellow_fingers_knn, anxiety_knn, 
                peer_pressure_knn, chronic_disease_knn, fatigue_knn, allergy_knn, 
                wheezing_knn, alcohol_knn, coughing_knn, breath_knn, swallowing_knn, 
                chest_pain_knn
            ]
            input_scaled_knn = scaler.transform(np.array(input_data_knn).reshape(1, -1))
            result_knn = knn_model.predict(input_scaled_knn)
            result_knn = "NO" if result_knn[0] == 1 else "YES"
            st.success(f"KNN Model predicts: Lung Cancer: {result_knn}")

if __name__ == '__main__':
    main()

