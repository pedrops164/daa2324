import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# import data
data = pd.read_csv('../input/train.csv')

# Split the data into training and test sets (80% train, 20% test)
train, test = train_test_split(data, test_size=0.2, random_state=2023)

train_X = train.drop('Price', axis=1) # Features (all columns except 'Price')
train_y = train['Price']

test_X = test.drop('Price', axis=1) # Features (all columns except 'Price')
test_y = test['Price']


def preprocess_name(train, test):
    # Preprocess the name column

    nbins = 9
    
    train['Brand'] = train['Name'].str.split().str[:2].str.join(' ') # Brand is the first two words of the Name column
    brand_avg_price_map_train = train.groupby('Brand')['Price'].transform('mean') # Average price for the brand of each entry
    # we store bin_edges so that we can assign the brands in the test set to their respective bin
    train['Brand_Bin'], bin_edges = pd.qcut(brand_avg_price_map_train, q=nbins, labels=range(nbins), duplicates='drop', retbins=True)

    # Compute the map between each brand and their average price in the training set
    brand_avg_price_map = train.groupby('Brand')['Price'].mean()

    # Map the average prices to the test set
    test_X_brand = test['Name'].str.split().str[:2].str.join(' ') # Brand is the first two words of the Name column
    brand_avg_price_map_test = test_X_brand.map(brand_avg_price_map)

    # Handle brands in the test set that were not seen in the training set
    # In this case, we'll fill NaN values with the median of the brand_avg_price_train
    # You can choose another strategy if it makes more sense for your use case
    brand_avg_price_map_test.fillna(brand_avg_price_map.median(), inplace=True)

    # Bin the test set using the bin edges from the training set
    test['Brand_Bin'] = pd.cut(brand_avg_price_map_test, bins=bin_edges, labels=range(len(bin_edges)-1), include_lowest=True)

    train.drop(['Brand', 'Name'], axis=1, inplace=True)
    test.drop(['Name'], axis=1, inplace=True)

    # might be even better to create one feature for each bin (one hot encoding!)

preprocess_name(train, test_X)