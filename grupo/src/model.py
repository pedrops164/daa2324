import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from preprocess import *
from predict import get_best_model
from sklearn.metrics import *
import numpy as np

# import data
data = pd.read_csv('../input/train.csv')

# Split the data into training and test sets (80% train, 20% test)
train, test = train_test_split(data, test_size=0.2, random_state=2023)

# tbag
preprocess_owner_type(train, test)
preprocess_seats(train, test)
preprocess_engine(train, test)

#hendrix
preprocess_power(train, test)
preprocess_new_price(train, test)

# garcon
preprocess_name(train, test)
preprocess_year(train, test)
preprocess_kilometers_driven(train, test)

#falta
preprocess_transmission(train, test)
preprocess_fuel_type(train, test)

train, test = preprocess_location(train, test)
train, test = preprocess_mileage(train, test)

print(train.head())
print(test.head())

print(train.info())
print()
print(test.info())

train_X = train.drop('Price', axis=1) # Features (all columns except 'Price')
train_y = train['Price']

test_X = test.drop('Price', axis=1) # Features (all columns except 'Price')
test_y = test['Price']

X = pd.concat([train_X, test_X], axis=0)

#get_best_model(train_X, train_y)
best_model, acc = get_best_model(train_X, train_y)

y_pred = best_model.predict(test_X)

mae = mean_absolute_error(test_y, y_pred)

mse = mean_squared_error(test_y, y_pred)

rmse = np.sqrt(mse)

print(f"MAE = {mae}\nRMSE = {rmse}")