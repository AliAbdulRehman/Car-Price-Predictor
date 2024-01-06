import pandas as pd
import joblib

# Load the model
loaded_model = joblib.load('car_predictor.joblib')
label_encoder_model = joblib.load('label_encoder_model.joblib')
label_encoder_brand = joblib.load('label_encoder_brand.joblib')
label_encoder_year = joblib.load('label_encoder_year.joblib')
scaler_price = joblib.load('scaler_price.joblib')

new_data = [{
    'brand': 'merc',  # Replace with actual values and column names
    'model': 'G Class',
    'year': 2020,
    'mileage': 1350,
    'fuel_economy': 21.4
}]
new_df = pd.DataFrame(new_data)

new_df['brand'] = label_encoder_brand.transform(new_df['brand'])
new_df['model'] = label_encoder_model.transform(new_df['model'])
new_df['year'] = label_encoder_year.transform(new_df['year'])  
prediction = loaded_model.predict(new_df)
print(prediction)
prediction_scaled = scaler_price.inverse_transform(prediction.reshape(-1, 1))

print(int(prediction_scaled[0][0]))

