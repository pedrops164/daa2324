import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from preprocess import preprocess_name, preprocess_year

# import data
data = pd.read_csv('../input/train.csv')

# Split the data into training and test sets (80% train, 20% test)
train, test = train_test_split(data, test_size=0.2, random_state=2023)

train_X = train.drop('Price', axis=1) # Features (all columns except 'Price')
train_y = train['Price']

test_X = test.drop('Price', axis=1) # Features (all columns except 'Price')
test_y = test['Price']

X = pd.concat([train_X, test_X], axis=0)

preprocess_name(train, test_X)
preprocess_year(X)