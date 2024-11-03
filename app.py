import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st

# Load the model
model = pk.load(open('model.pkl', 'rb'))

# Streamlit header
st.header('Car Price Prediction ML Model')

# Load and preprocess the car data
cars_data = pd.read_csv('car data.csv')

def get_brand_name(Car_name):
    Car_name = Car_name.split(' ')[0]
    return Car_name.strip(' ')

cars_data['Car_Name'] = cars_data['Car_Name'].apply(get_brand_name)

# Streamlit input fields
Car_Name = st.selectbox('Select Car Brand', cars_data['Car_Name'].unique())
Year = st.slider('Car Manufactured Year', 2003, 2018)
Driven_kms = st.slider('No of kms Driven', 1, 500000)
Present_Price = st.slider('Present Price', 0.32, 92.6)
Fuel_Type = st.selectbox('Fuel type', cars_data['Fuel_Type'].unique())
Selling_type = st.selectbox('Seller type', cars_data['Selling_type'].unique())
Transmission = st.selectbox('Transmission type', cars_data['Transmission'].unique())
Owner = st.selectbox('Owner Count', cars_data['Owner'].unique())

# Prediction
if st.button("Predict"):
    # Create a DataFrame for the model input
    Sample_data = pd.DataFrame(
        [[Car_Name, Year, Present_Price, Driven_kms, Fuel_Type, Selling_type, Transmission, Owner]],
        columns=['Car_Name', 'Year', 'Present_Price', 'Driven_kms', 'Fuel_Type', 'Selling_type', 'Transmission', 'Owner']
    )
    
    # Convert categorical columns to numeric as required by the model
    Sample_data['Car_Name'].replace(
        ['ritz', 'sx4', 'ciaz', 'wagon', 'swift', 'vitara', 's', 'alto', 'ertiga', 'dzire', 'ignis', '800', 
         'baleno', 'omni', 'fortuner', 'innova', 'corolla', 'etios', 'camry', 'land', 'Royal', 'UM', 
         'KTM', 'Bajaj', 'Hyosung', 'Mahindra', 'Honda', 'Yamaha', 'TVS', 'Hero', 'Activa', 'Suzuki', 
         'i20', 'grand', 'i10', 'eon', 'xcent', 'elantra', 'creta', 'verna', 'city', 'brio', 'amaze', 'jazz'], 
        list(range(1, 45)), inplace=True
    )
    Sample_data['Fuel_Type'].replace(['Petrol', 'Diesel', 'CNG'], [1, 2, 3], inplace=True)
    Sample_data['Selling_type'].replace(['Dealer', 'Individual'], [1, 2], inplace=True)
    Sample_data['Transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    
    # Predict the car price
    car_price = model.predict(Sample_data)
    
    # Display the result
    st.markdown('Car Price is going to be ' + str(car_price[0]))
