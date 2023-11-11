import pandas as pd
from sklearn.preprocessing import LabelBinarizer

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
    lb = LabelBinarizer()
    lb_resultsTrain = lb.fit_transform(train_X['Location'])
    lb_resultsTest = lb.fit_transform(test_X['Location'])
    resTrain = pd.DataFrame(lb_resultsTrain, columns=lb.classes_)
    resTest = pd.DataFrame(lb_resultsTest, columns=lb.classes_)
    pd.concat([train_X, resTrain], axis=1)
    pd.concat([test_X, resTest], axis=1)

def preprocess_year(train_X, test_X):

    # Determine the maximum year from both train and test sets to use as a reference
    max_year = max(train_X['Year'].max(), test_X['Year'].max())

    # Group years before 2006 for both train and test sets
    train_X['Year'] = train_X['Year'].apply(lambda x: 2005 if x <= 2005 else x)
    test_X['Year'] = test_X['Year'].apply(lambda x: 2005 if x <= 2005 else x)

    # Subtract the max year from the year column to represent the age of the car
    train_X['Age'] = max_year - train_X['Year']
    test_X['Age'] = max_year - test_X['Year']

    # Drop the Year column as it's no longer needed
    train_X.drop(['Year'], inplace=True, axis=1)
    test_X.drop(['Year'], inplace=True, axis=1)

    # We can also one hot encode the Age instead, might be better
    # train_X = pd.get_dummies(train_X, columns=['Age'], prefix='Age', drop_first=True)
    # test_X = pd.get_dummies(test_X, columns=['Age'], prefix='Age', drop_first=True)

def preprocess_kilometers_driven(train, test):
    from sklearn.preprocessing import StandardScaler

    # We scale the kilometers driven so that the feature has a mean of 0 and std deviation of 1 (z-score normalization)
    scaler = StandardScaler()
    train['Kilometers_Driven'] = scaler.fit_transform(train[['Kilometers_Driven']])
    test['Kilometers_Driven'] = scaler.fit_transform(test[['Kilometers_Driven']])

    train.rename(columns={'Kilometers_Driven': 'Km_Driven_Scaled'}, inplace=True)
    test.rename(columns={'Kilometers_Driven': 'Km_Driven_Scaled'}, inplace=True)

def preprocess_fuel_type(train_X, test_X):
    replace_map = {'Fuel_Type': { 'Diesel': 1, 'Petrol': 2, 'Eletric': 3}}
    train_X.replace(replace_map, inplace=True)
    test_X.replace(replace_map, inplace=True)

def preprocess_transmission(train_X, test_X):
    replace_map2 = {'Transmission': { 'Manual': 1, 'Automatic': 2}}
    train_X.replace(replace_map2, inplace=True)
    test_X.replace(replace_map2, inplace=True)


def preprocess_owner_type(train_X, test_X):
    # There are only 4 values, 'First', 'Second', 'Third' and 'Fourth & Above', so we create a replace map, joining the 2 last ones
    # as they have a small number of occurences
    replace_map = {'Owner_Type': { 'First': 1, 'Second': 2, 'Third': 3, 'Fourth & Above':3 }}
    
    train_X.replace(replace_map, inplace=True)
    test_X.replace(replace_map, inplace=True)

def preprocess_mileage(train_X, test_X):
    MileageKMPL = []

    for index,row in train_X.iterrows():
        m = row['Mileage']
        if 'km/kg' in m:
            m = m[:-6]
            if 1==row['Fuel_Type']:
                m = float(m)*0.84
                MileageKMPL.append(float(m))
            else:
                m = float(m)*0.75
                MileageKMPL.append(float(m))
        else:
            m = m[:-5]
            MileageKMPL.append(float(m))
        pass

    MileageKMPLTest = []

    for index,row in test_X.iterrows():
        m = row['Mileage']
        if 'km/kg' in m:
            m = m[:-6]
            if 1==row['Fuel_Type']:
                m = float(m)*0.84
                MileageKMPLTest.append(float(m))
            else:
                m = float(m)*0.75
                MileageKMPLTest.append(float(m))
        else:
            m = m[:-5]
            MileageKMPLTest.append(float(m))
        pass

    train_X['Mileage']=MileageKMPL
    test_X['Mileage']=MileageKMPLTest

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

    # observed=True to supress the deprecated warnings
    groupByPriceBinTrain = train_X.groupby(priceBinTrain, observed=True)['Engine'].mean().to_dict()
    groupByPriceBinTest = test_X.groupby(priceBinTest, observed=True)['Engine'].mean().to_dict()
    
    train_X['Engine'].fillna(train_X['Price'].map(groupByPriceBinTrain), inplace=True)
    test_X['Engine'].fillna(test_X['Price'].map(groupByPriceBinTest), inplace=True)

    # Check for any remaining NaN values and fill them with a default value or handle them as needed
    # We will fill with the overall mean of the 'Engine' column
    # Calculate the mean engine value for the whole training set
    overall_engine_mean = train_X['Engine'].mean()
    train_X['Engine'].fillna(overall_engine_mean, inplace=True)
    test_X['Engine'].fillna(overall_engine_mean, inplace=True)

    train_X['Engine'] = train_X['Engine'].astype(int)
    test_X['Engine'] = test_X['Engine'].astype(int)


def preprocess_power(train_X, test_X):
    # TODO
    def removeBHP(str_val):
        parts = str_val.split(" bhp")
        if parts[0]:  # Check if the first part is not an empty string
            return float(parts[0])
        else:
            return None  # Return None if there's no numeric part before " bhp"


    train_X['Power'] = train_X['Power'].map(removeBHP, na_action="ignore")
    test_X['Power'] = test_X['Power'].map(removeBHP, na_action="ignore")
    
    groupNamePowerTrain = train_X.groupby('Name')['Power'].mean().to_dict()
    groupNamePowerTest = test_X.groupby('Name')['Power'].mean().to_dict()
    
    train_X['Power'].fillna(train_X['Name'].map(groupNamePowerTrain), inplace=True)
    test_X['Power'].fillna(test_X['Name'].map(groupNamePowerTest), inplace=True)
    
    train_X.dropna(subset=['Power'], inplace=True)
    test_X.dropna(subset=['Power'], inplace=True)
    
def preprocess_seats(train_X, test_X):
    # TODO
    groupSitsByNameTrain = train_X.groupby('Name')['Seats'].mean().to_dict()
    groupSitsByNameTest = test_X.groupby('Name')['Seats'].mean().to_dict()
    
    train_X['Seats'].fillna(train_X['Name'].map(groupSitsByNameTrain), inplace=True)
    test_X['Seats'].fillna(test_X['Name'].map(groupSitsByNameTest), inplace=True)
    
    train_X['Seats'] = train_X['Seats'].fillna(train_X['Seats'].mode()[0])
    test_X['Seats'] = test_X['Seats'].fillna(test_X['Seats'].mode()[0])
    

def preprocess_new_price(train_X, test_X):
    # TODO
    train_X.drop(columns=['New_Price'], axis=1, inplace=True)
    test_X.drop(columns=['New_Price'], axis=1, inplace=True)