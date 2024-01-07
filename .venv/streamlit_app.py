import streamlit as st
import pandas as pd
import joblib


df = pd.read_csv('.venv\combined_filtered2.csv')

def predict_price(brand, model, year, mileage, fuel_economy):

    loaded_model = joblib.load('car_predictor.joblib')
    label_encoder_model = joblib.load('label_encoder_model.joblib')
    label_encoder_brand = joblib.load('label_encoder_brand.joblib')
    label_encoder_year = joblib.load('label_encoder_year.joblib')
    scaler_price = joblib.load('scaler_price.joblib')
    new_data = [{
    'brand': brand,  
    'model': model,
    'year': year,
    'mileage': mileage,
    'fuel_economy': fuel_economy
    }]
    new_df = pd.DataFrame(new_data)

    new_df['brand'] = label_encoder_brand.transform(new_df['brand'])
    new_df['model'] = label_encoder_model.transform(new_df['model'])
    new_df['year'] = label_encoder_year.transform(new_df['year'])  
    prediction = loaded_model.predict(new_df)
    prediction_scaled = scaler_price.inverse_transform(prediction.reshape(-1, 1))

    predicted_price = int(prediction_scaled[0][0])
    return predicted_price


st.title('Car Price Predictor')
st.write('Choose the parameters of the car whose price you want to predict!')

car_brand = st.selectbox('Select Car Brand', options=df['brand'].unique())

if car_brand:
    models = df[df['brand'] == car_brand]['model'].unique()
    car_model = st.selectbox('Select Car Model', options=models)

    if car_model:
        year = st.slider('Select Year', min_value=1996, max_value=2010, value=2005)

      
        mileage = st.number_input('Enter a number for Mileage', min_value=1, max_value=10000)
        fuel_economy = st.number_input('Enter a number for Fuel Economy', min_value=1, max_value=10000)

   
        if st.button('Predict Price'):
     
            predicted_price = predict_price(car_brand, car_model, year, mileage, fuel_economy)
            st.write(f'The predicted price is: £{predicted_price}')