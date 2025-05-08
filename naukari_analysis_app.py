import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load LabelEncoders using pickle
with open('le_company.pkl', 'rb') as f:
    le_company = pickle.load(f)

with open('le_location.pkl', 'rb') as f:
    le_location = pickle.load(f)

with open('randomtree.pkl', 'rb') as f:
    randomtree = pickle.load(f)
# Load the trained XGBoost model
#randomtree = joblib.load('naukri_analysis.pkl')
#encoder_loc = joblib.load('le_location.pkl')
#encoder_comp = joblib.load('le_company.pkl')
# Title and Description
st.title('Naukri Job Analysis')
st.write('Enter your skills, experience, location, rating, and company to predict your best job match!')

# Skills input
Reviews=st.number_input("enter number of reviews : ",min_value=0,max_value=100000,value=0)

# Experience input
experience_minimum = st.number_input('Minimum experience:', min_value=0, max_value=12, value=0)
experience_maximum = st.number_input('Maximum experience:', min_value=0, max_value=15, value=0)
Experience_Avg = (experience_minimum + experience_maximum) / 2.0

# Location input
location_options = ['Mumbai',
       'Mumbai (All Areas), Hyderabad/Secunderabad, Pune, Chennai, Delhi / NCR, Bangalore/Bengaluru',
       'Mumbai, Gurgaon/Gurugram, Aurangabad, Vadodara',
       'Mumbai (All Areas)',
       'Pune, Hyderabad/Secunderabad, Chennai, Delhi / NCR, Bangalore/Bengaluru, Mumbai (All Areas)',
       'Pune',
       'Chennai, Hyderabad/Secunderabad, Pune, Delhi / NCR, Bangalore/Bengaluru, Mumbai (All Areas)',
       'Chennai', 'Chennai(Ekkaduthangal)',
       'Chennai(Kodambakkam), Kodambakkam']
Location = st.selectbox("Select your location:", location_options)

# Rating input
Ratings = st.number_input('Your rating (out of 5)', min_value=0.0, max_value=5.0, value=3.0,step = 1.0)


# Company input
company_options = ['Accenture', 'CoinDCX', 'Oracle', 'Siemens', 'Rave Technologies',
       'HealthSpring', 'Citibank, N.A', 'Snaphunt', 'Duff & Phelps',
       'Prodair Air Products', 'Air Products', 'CompuCom',
       'Method Studios', 'Company3 Method India Private Limited',
       'Thinksynq Solutions', 'Shell', 'Icon Clinical Research',
       'Aspire Systems', 'Icon Pharmaceutical s',
       'Associated Auto Solutions International Pvt. Ltd.',
       'Sona Comstar', 'NatWest Group', 'Eversendai']
Company = st.selectbox('Select your company:', company_options)

if st.button('Predict'):
    # Prepare the features for prediction
    
    # Encode Location
    Location_enc = le_location.transform([Location])[0]

# Encode Company
    Company_enc = le_company.transform([Company])[0]


# Combine all features into a single array for prediction
    features = np.array([[Experience_Avg, Reviews, Ratings, Company_enc, Location_enc]])

 
    # Make prediction
    prediction = randomtree.predict(features)
    st.success(f'The best job match for you is: {prediction[0]}')

    # Store prediction for plotting if needed
    st.session_state.prediction = prediction

if st.button("Show Predictions Plot"):
    if 'prediction' in st.session_state:
        plt.figure(figsize=(10, 6))
        plt.plot(st.session_state.prediction, 'bo', label='Prediction')
        plt.xlabel('Samples')
        plt.ylabel('Job Match Score')
        plt.title('Prediction of best available job(title)')
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("Please make a prediction first by clicking the 'Predict' button")

