import pandas as pd

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

def preprocess_location(train_X, test_X):
    # TODO
    pass

def preprocess_year(X):

    # We Group years before 2006
    X['Year'] = X['Year'].apply(lambda x: 2005 if x <= 2005 else x)

    # We can subtract the max year from the year column, so that the values in the year column represent the age of the car
    X['Age'] = X['Year'].max() - X['Year']

    # We drop the Year column because we don't need it anymore
    X.drop(['Year'], inplace=True, axis=1)

    # We can also one hot encode the Age instead, might be better
    # X = pd.get_dummies(X, columns=['Age'], prefix='Age', drop_first=True)

def preprocess_kilometers_driven(X):
    from sklearn.preprocessing import StandardScaler

    # We scale the kilometers driven so that the feature has a mean of 0 and std deviation of 1 (z-score normalization)
    scaler = StandardScaler()
    X['Kilometers_Driven'] = scaler.fit_transform(X[['Kilometers_Driven']])

def preprocess_fuel_type(train_X, test_X):
    # TODO
    pass

def preprocess_transmission(train_X, test_X):
    # TODO
    pass

def preprocess_owner_type(train_X, test_X):
    # There are only 4 values, 'First', 'Second', 'Third' and 'Fourth & Above', so we create a replace map, joining the 2 last ones
    # as they have a small number of occurences
    replace_map = {'Owner_Type': { 'First': 1, 'Second': 2, 'Third': 3, 'Fourth & Above':3 }}
    
    train_X.replace(replace_map, inplace=True)
    test_X.replace(replace_map, inplace=True)

def preprocess_mileage(train_X, test_X):
    # TODO
    pass

def preprocess_engine(train_X, test_X):
    # TODO
    # Function to remove CC from Engine and transform to int
    def removeCC(val):
        num, _ = val.split(" ")
        return float(num)

    train_X['Engine'] = train_X['Engine'].map(removeCC, na_action="ignore")
    test_X['Engine'] = test_X['Engine'].map(removeCC, na_action="ignore")
    
    groupByNameTrain = train_X.groupby('Name')['Engine'].mean().to_dict()
    groupByNameTest = test_X.groupby('Name')['Engine'].mean().to_dict()
    
    train_X['Engine'].fillna(train_X['Name'].map(groupByNameTrain), inplace=True)
    test_X['Engine'].fillna(test_X['Name'].map(groupByNameTest), inplace=True)
    
    priceBinTrain = pd.cut(train_X['Price'], bins=int(train_X['Price'].max()))
    priceBinTest = pd.cut(test_X['Price'], bins=int(test_X['Price'].max()))

    groupByPriceBinTrain = train_X.groupby(priceBinTrain)['Engine'].mean().to_dict()
    groupByPriceBinTest = test_X.groupby(priceBinTest)['Engine'].mean().to_dict()
    
    train_X['Engine'].fillna(train_X['Price'].map(groupByPriceBinTrain), inplace=True)
    train_X['Engine'] = train_X['Engine'].astype(int)
    test_X['Engine'].fillna(test_X['Price'].map(groupByPriceBinTest), inplace=True)
    test_X['Engine'] = test_X['Engine'].astype(int)

def preprocess_power(train_X, test_X):
    # TODO
    pass

def preprocess_seats(train_X, test_X):
    # TODO
    pass

def preprocess_new_price(train_X, test_X):
    # TODO
    # just drop new_price column
    pass