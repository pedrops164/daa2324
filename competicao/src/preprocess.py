import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer, MinMaxScaler
from miceforest import ImputationKernel
from util import random_state
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import numpy as np

# Rename columns

def rename_energia(energia_train, energia_test_X):
    energia_train.rename(columns={"Data": "data", "Hora": "hora", "Normal (kWh)": "normal", "Horário Económico (kWh)": "horario",
                               "Autoconsumo (kWh)": "autoconsumo", "Injeção na rede (kWh)": "injecao"}, inplace=True)
    energia_test_X.rename(columns={"Data": "data", "Hora": "hora", "Normal (kWh)": "normal", "Horário Económico (kWh)": "horario",
                               "Autoconsumo (kWh)": "autoconsumo", "Injeção na rede (kWh)": "injecao"}, inplace=True)

# Merge datasets

# merges the energia and meteo datasets, into train and test set
def merge_by_date(train_energia_df, train_meteo_df, test_energia_df, test_meteo_df):
    def remove_utc(s):
        if 'UTC' in s:
            return s.replace(" +0000 UTC", "")
        return s
    
    # remove UTC from the end and transform to datetime
    train_meteo_df['dt_iso'] = pd.to_datetime(train_meteo_df['dt_iso'].map(remove_utc))
    test_meteo_df['dt_iso'] = pd.to_datetime(test_meteo_df['dt_iso'].map(remove_utc))
    
    # extract date and hour to different columns
    train_meteo_df['data'] = train_meteo_df['dt_iso'].dt.date
    train_meteo_df['ano'] = train_meteo_df['dt_iso'].dt.year
    train_meteo_df['mes'] = train_meteo_df['dt_iso'].dt.month
    train_meteo_df['dia'] = train_meteo_df['dt_iso'].dt.day
    train_meteo_df['hora'] = train_meteo_df['dt_iso'].dt.hour
    
    test_meteo_df['data'] = test_meteo_df['dt_iso'].dt.date
    test_meteo_df['ano'] = test_meteo_df['dt_iso'].dt.year
    test_meteo_df['mes'] = test_meteo_df['dt_iso'].dt.month
    test_meteo_df['dia'] = test_meteo_df['dt_iso'].dt.day
    test_meteo_df['hora'] = test_meteo_df['dt_iso'].dt.hour

    # create year month and day columns for energia dataset
    train_energia_datetime = pd.to_datetime(train_energia_df['data'])
    train_energia_df['data'] = train_energia_datetime.dt.date
    train_energia_df['ano'] = train_energia_datetime.dt.year
    train_energia_df['mes'] = train_energia_datetime.dt.month
    train_energia_df['dia'] = train_energia_datetime.dt.day

    test_energia_datetime = pd.to_datetime(test_energia_df['data'])
    test_energia_df['data'] = test_energia_datetime.dt.date
    test_energia_df['ano'] = test_energia_datetime.dt.year
    test_energia_df['mes'] = test_energia_datetime.dt.month
    test_energia_df['dia'] = test_energia_datetime.dt.day

    # Merge based on the list of columns: 'ano', 'mes', 'dia', and 'hora'
    merge_columns = ['data', 'ano', 'mes', 'dia', 'hora']

    #return the merged datasets
    # the join in the test dataset has to be outer, and we gotta predict the missing values in the meteo dataset.
    return  pd.merge(train_energia_df, train_meteo_df, how='inner', on=merge_columns), \
            pd.merge(test_energia_df, test_meteo_df, how='outer', on=merge_columns) 

# fill the missing values
def fill_missing_values(test_X):
    mice_kernel = ImputationKernel(
    data = test_X,
    save_all_iterations = True,
    random_state = random_state)

    mice_kernel.mice(3)
    test_mice_imputation = mice_kernel.complete_data()
    return test_mice_imputation

# examples implementation
def implementation_samples(X_train, y_train):
    smote = SMOTE(sampling_strategy={1: 1000, 2: 1000, 3: 1000, 4: 1000})
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def overundersampling(X_train, y_train):
    over = SMOTE(sampling_strategy={1: 1000, 2: 1000, 3: 1000, 4: 1000})  # or 'auto' for naive over-sampling
    under = RandomUnderSampler(sampling_strategy={0: 3000})  # or 'auto' for naive under-sampling

    # Create a pipeline that first oversamples then undersamples
    pipeline = Pipeline(steps=[('o', over), ('u', under)])

    # Transform the dataset
    X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


# PREPROCESSAMENTO ENERGIA

# preprocessing of dates
def preprocess_dates(train, test_X):
    #train['weekday'] = pd.to_datetime(train['data']).dt.weekday
    #train['monthday'] = pd.to_datetime(train['data']).dt.day
    #test_X['weekday'] = pd.to_datetime(test_X['data']).dt.weekday
    #test_X['monthday'] = pd.to_datetime(test_X['data']).dt.day

    train['day_of_year'] = pd.to_datetime(train['data']).dt.dayofyear
    test_X['day_of_year'] = pd.to_datetime(test_X['data']).dt.dayofyear

    train.drop(["dt"], inplace=True, axis=1)
    train.drop(["dt_iso"], inplace=True, axis=1)
    train.drop(["data"], inplace=True, axis=1)
    train.drop(["ano"], inplace=True, axis=1)
    train.drop(["mes"], inplace=True, axis=1)
    train.drop(["dia"], inplace=True, axis=1)
    test_X.drop(["dt"], inplace=True, axis=1)
    test_X.drop(["dt_iso"], inplace=True, axis=1)
    test_X.drop(["data"], inplace=True, axis=1)
    test_X.drop(["ano"], inplace=True, axis=1)
    test_X.drop(["mes"], inplace=True, axis=1)
    test_X.drop(["dia"], inplace=True, axis=1)

def preprocess_hora(train, test_X):
    pass

def preprocess_normal(train, test_X):
    pass

def preprocess_horario(train, test_X):
    # horario economico
    pass

def preprocess_autoconsumo(train, test_X):
    pass

def preprocess_injecao(train, test_X):
    # injecao na rede
    order_mapping = {'None': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
    train['injecao'] = train['injecao'].map(order_mapping)
    #le = LabelEncoder()
    #train['injecao'] = le.fit_transform(train['injecao'])

    # we one hot encode the categorical variable
    #ohe_train = pd.get_dummies(train['injecao'], prefix='injecao')
    #train = train.join(ohe_train)
    #train.drop(['injecao'], axis=1, inplace=True)
    #return train, test_X


def fe_energia(train, test_X):
    # criacao de features compostas
    train['horario_autoconsumo'] = train['horario'] * train['autoconsumo']
    test_X['horario_autoconsumo'] = test_X['horario'] * test_X['autoconsumo']

    train['normal_horario'] = train['normal'] * train['horario']
    test_X['normal_horario'] = test_X['normal'] * test_X['horario']
    
    train['hora_horario'] = train['hora'] * train['horario']
    test_X['hora_horario'] = test_X['hora'] * test_X['horario']

    train['hora_normal'] = train['hora'] * train['normal']
    test_X['hora_normal'] = test_X['hora'] * test_X['normal']


# PREPROCESSAMENTO METEO

def preprocess_city_name(train, test_X):
    # all entries have the same value, so we can drop the column
    train.drop(["city_name"], inplace=True, axis=1)
    test_X.drop(["city_name"], inplace=True, axis=1)

def preprocess_temp(train, test_X):
    # Fill missing values with mean
    #temp_mean = train['temp'].mean()
    #train['temp'].fillna(temp_mean, inplace=True)
    #test_X['temp'].fillna(temp_mean, inplace=True)
    train['temp_diff'] = train['temp_max'] - train['temp_min']
    test_X['temp_diff'] = test_X['temp_max'] - test_X['temp_min']

def preprocess_feelslike(train, test_X):
    # Fill missing values with mean
    #feels_like_mean = train['feels_like'].mean()
    #train['feels_like'].fillna(feels_like_mean, inplace=True)
    #test_X['feels_like'].fillna(feels_like_mean, inplace=True)
    pass

def preprocess_temp_min(train, test_X):
    # Fill missing values with mean
    #temp_min_mean = train['temp_min'].mean()
    #train['temp_min'].fillna(temp_min_mean, inplace=True)
    #test_X['temp_min'].fillna(temp_min_mean, inplace=True)
    pass

def preprocess_temp_max(train, test_X):
    # Fill missing values with mean
    #temp_max_mean = train['temp_max'].mean()
    #train['temp_max'].fillna(temp_max_mean, inplace=True)
    #test_X['temp_max'].fillna(temp_max_mean, inplace=True)
    pass

def preprocess_pressure(train, test_X):
    # Fill missing values with median
    #pressure_median = train['pressure'].median()
    #train['pressure'].fillna(pressure_median, inplace=True)
    #test_X['pressure'].fillna(pressure_median, inplace=True)
    pass

def preprocess_sea_level(train, test_X):
    # all entries are null values
    train.drop(["sea_level"], inplace=True, axis=1)
    test_X.drop(["sea_level"], inplace=True, axis=1)

def preprocess_grnd_level(train, test_X):
    # all entries are null values
    train.drop(["grnd_level"], inplace=True, axis=1)
    test_X.drop(["grnd_level"], inplace=True, axis=1)

def preprocess_humidity(train, test_X):
    # Fill missing values with mean
    #humidity_mean = train['humidity'].mean()
    #train['humidity'].fillna(humidity_mean, inplace=True)
    #test_X['humidity'].fillna(humidity_mean, inplace=True)
    pass

def preprocess_wind_speed(train, test_X):
    # Fill missing values with mean
    #wind_speed_mean = train['wind_speed'].mean()
    #train['wind_speed'].fillna(wind_speed_mean, inplace=True)
    #test_X['wind_speed'].fillna(wind_speed_mean, inplace=True)
    pass

def preprocess_rain1h(train, test_X):
    # Maybe remove? A lot of null values
    train.drop(["rain_1h"], inplace=True, axis=1)
    test_X.drop(["rain_1h"], inplace=True, axis=1)

def preprocess_clouds_all(train, test_X):
    # Fill missing values with mean
    #clouds_all_mean = train['clouds_all'].mean()
    #train['clouds_all'].fillna(clouds_all_mean, inplace=True)
    #test_X['clouds_all'].fillna(clouds_all_mean, inplace=True)
    pass

def preprocess_weather_description(train, test_X):
    # since the weather_description column only has 8 possible values, one hot encode it

    # we convert the boolean to floats
    ohe_train = pd.get_dummies(train['weather_description'], prefix='wd', dummy_na=True).astype(float)
    train = train.join(ohe_train)

    ohe_test = pd.get_dummies(test_X['weather_description'], prefix='wd', dummy_na=True).astype(float)
    test_X = test_X.join(ohe_test)

    # Align train and test sets to ensure they have the same dummy columns
    #train, test_X = train.align(test_X, join='outer', axis=1, fill_value=0)

    # Find rows where 'weather_description' was NaN and set all dummy columns to NaN for those rows
    train.loc[train['weather_description'].isna(), ohe_train.columns] = np.nan
    test_X.loc[test_X['weather_description'].isna(), ohe_test.columns] = np.nan

    train.drop("weather_description", axis=1, inplace=True)
    test_X.drop("weather_description", axis=1, inplace=True)

    return train, test_X

def scale_features(train, test_X):
    features=['hora','normal','horario','autoconsumo','temp','feels_like','temp_min','temp_max','pressure',
              'humidity','wind_speed','clouds_all','day_of_year','temp_diff']
    for feature in features:
        # quantile transformer
        #scaler = QuantileTransformer(output_distribution="normal", random_state=random_state)
        # minmax scaling
        #scaler = MinMaxScaler()
        # standard scaling
        scaler = StandardScaler()

        # Fit the scaler to the train data and transform train data
        train[feature] = scaler.fit_transform(train[[feature]])

        # Transform test data using the same scaler
        test_X[feature] = scaler.transform(test_X[[feature]])

def remove_outliers(train):
    cols = ['normal','horario','autoconsumo','temp','feels_like','temp_min','temp_max','humidity','wind_speed','clouds_all',
            'temp_diff']

    # Define a dictionary to hold the outlier indices for each column
    outlier_indices = {}

    # Loop over each column in the DataFrame
    for column in cols:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile) for the given column
        Q1 = train[column].quantile(0.25)
        Q3 = train[column].quantile(0.75)
        # Calculate the IQR (Interquartile Range)
        IQR = Q3 - Q1
        # Define the range for outliers as 1.5 times the IQR from the Q1 and Q3
        outlier_step = 1.5 * IQR
        # Find the indices of outliers in the column and add them to the dictionary
        outlier_list_col = train[(train[column] < Q1 - outlier_step) | (train[column] > Q3 + outlier_step)].index
        outlier_indices[column] = outlier_list_col

    # Find the list of unique indices that have outliers in more than one column
    # This step is optional and depends on whether you want to be strict about outlier removal
    outliers = []
    for idx_list in outlier_indices.values():
        for idx in idx_list:
            if idx not in outliers:
                outliers.append(idx)

    print(len(outliers))
    # Drop the outliers and return the DataFrame without outliers
    df_cleaned = train.drop(outliers)
    return df_cleaned

# Achieves slightly better results
def remove_outliers_isolation_forest(train, contamination_factor=0.01):
    cols = ['normal','horario','autoconsumo','temp','feels_like','temp_min','temp_max','humidity','wind_speed','clouds_all',
            'temp_diff']
    
    # We will collect the indices of rows considered inliers by IsolationForest for all specified columns
    inlier_indices = []

    for col in cols:
        # Initialize the IsolationForest model
        iso_forest = IsolationForest(contamination=contamination_factor, random_state=42)

        # Reshape the data to fit the model: it should be 2D (samples, features)
        col_data = train[col].values.reshape(-1, 1)
        
        # Fit the model
        iso_forest.fit(col_data)

        # Predict the anomalies
        preds = iso_forest.predict(col_data)

        # Store the indices of inliers (where preds == 1)
        inlier_indices.append(train[col][preds == 1].index)

    # Find the intersection of all inlier indices
    inliers_common = set(inlier_indices[0])
    for indices in inlier_indices[1:]:
        inliers_common.intersection_update(indices)

    # Convert the set of inlier indices to a list
    inliers_common = list(inliers_common)

    # Return the dataframe with only the inliers
    return train.loc[inliers_common].reset_index(drop=True)
