import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler 
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import joblib

df = pd.read_csv("./combined_filtered2.csv")
df['brand'] = df['brand'].astype('string')
df['model'] = df['model'].astype('string')
##########################################################
max_val = np.max(df[["price"]],axis =0)
min_val = np.min(df[["price"]],axis =0)
avg_val = np.mean((df[["price"]]),axis =0)
print(df.dtypes)
#converting categorical columns to numerical columns so that they can be used in the ML algorithms
label_encoder_model = LabelEncoder()
df['model'] = label_encoder_model.fit_transform(df['model'])

label_encoder_year = LabelEncoder()
df['year'] = label_encoder_year.fit_transform(df['year'])

label_encoder_brand = LabelEncoder()
df['brand'] = label_encoder_brand.fit_transform(df['brand'])

joblib.dump(label_encoder_model, 'label_encoder_model.joblib')
joblib.dump(label_encoder_brand, 'label_encoder_brand.joblib')
joblib.dump(label_encoder_year, 'label_encoder_year.joblib')
# normalizing data between 0 and 1 to transform features to be on a similar scale which will improve the performance
scaler_price = MinMaxScaler()
scaler_mileage = MinMaxScaler()
scaler_year = MinMaxScaler()
df[["price"]] = scaler_price.fit_transform(df[["price"]])
df[['mileage']] = scaler_mileage.fit_transform(df[["mileage"]])
# df[['year']] = scaler_year.fit_transform(df[["year"]])

joblib.dump(scaler_price, 'scaler_price.joblib')

y = df["price"] #target variable

#creating the feature matrix and dropping all columns which are not correlated or dependant
X = df.drop(['price'],axis=1)
X = X.dropna()
print(X.dtypes)
#splitting the training, validation and testing data 60-20-20
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

#using the random forest regression algorithm on the training data
rf_regressor = RandomForestRegressor(n_estimators=100,random_state=42)
rf_regressor.fit(X_train, y_train)


model_dump = 'car_predictor.joblib'
joblib.dump(rf_regressor, model_dump)


# y_pred_train_rf = rf_regressor.predict(X_train)

# #converting the values back to the original scale (inverse of normalization)
# y_pred_train_rf_orig = scaler1.inverse_transform(y_pred_train_rf.reshape(1,-1))
# y_train_orig = scaler1.inverse_transform(y_train.values.reshape(1,-1))

# #data transformation
# y_pred_train_rf_orig = np.array(y_pred_train_rf_orig).flatten()
# y_train_orig = np.array(y_train_orig).flatten()

# #calculating the mean absolute error
# mae_train_rf = mean_absolute_error(y_pred_train_rf_orig,y_train_orig)
# print("training error: ",(mae_train_rf/avg_val)*100)

# ###################################################################
# #using the random forest regression algorithm on the validation data
# y_val_pred_rf = rf_regressor.predict(X_val)

# #converting the values back to the original scale (inverse of normalization)
# y_val_pred_rf_orig = scaler1.inverse_transform(y_val_pred_rf.reshape(1,-1))
# y_val_orig = scaler1.inverse_transform(y_val.values.reshape(1,-1))

# #data transformation
# y_val_pred_rf_orig = np.array(y_val_pred_rf_orig).flatten()
# y_val_orig = np.array(y_val_orig).flatten()

# #calculating the mean absolute error
# mae_rf = mean_absolute_error(y_val_orig,y_val_pred_rf_orig)
# print("validation error", (mae_rf/avg_val)*100)


# #using the random forest regression algorithm on the testing data as the validation data produced better results for random forest
# y_pred = rf_regressor.predict(X_test)

# #converting the values back to the original scale (inverse of normalization)
# y_pred_orig = scaler1.inverse_transform(y_pred.reshape(1,-1))
# y_orig = scaler1.inverse_transform(y_test.values.reshape(1,-1))

# #data transformation
# y_orig = np.array(y_orig).flatten()
# y_pred_orig = np.array(y_pred_orig).flatten()

# #calculating the mean absolute error
# mae_test = mean_absolute_error(y_orig,y_pred_orig)
# print((mae_test/avg_val)*100)