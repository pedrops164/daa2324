import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer
from miceforest import ImputationKernel
from util import random_state
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

# PREPROCESSAMENTO ENERGIA

# preprocessing of dates
def preprocess_dates(train, test_X):
    #train['weekday'] = pd.to_datetime(train['data']).dt.weekday
    #train['monthday'] = pd.to_datetime(train['data']).dt.day
    #test_X['weekday'] = pd.to_datetime(test_X['data']).dt.weekday
    #test_X['monthday'] = pd.to_datetime(test_X['data']).dt.day

    train.drop(["dt"], inplace=True, axis=1)
    train.drop(["dt_iso"], inplace=True, axis=1)
    train.drop(["data"], inplace=True, axis=1)
    test_X.drop(["dt"], inplace=True, axis=1)
    test_X.drop(["dt_iso"], inplace=True, axis=1)
    test_X.drop(["data"], inplace=True, axis=1)

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
    pass


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
    pressure_median = train['pressure'].median()
    train['pressure'].fillna(pressure_median, inplace=True)
    test_X['pressure'].fillna(pressure_median, inplace=True)

    # z score standardization
    # Create a StandardScaler object
    #scaler = StandardScaler()

    # quantile transformer
    scaler = QuantileTransformer(output_distribution="normal", random_state=random_state)

    # Fit the scaler to the train data and transform train data
    train['pressure'] = scaler.fit_transform(train[['pressure']])

    # Transform test data using the same scaler
    test_X['pressure'] = scaler.transform(test_X[['pressure']])


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

    # z score standardization
    # Create a StandardScaler object
    #scaler = StandardScaler()

    # quantile transformer
    scaler = QuantileTransformer(output_distribution="normal", random_state=random_state)
    
    # Fit the scaler to the train data and transform train data
    train['humidity'] = scaler.fit_transform(train[['humidity']])

    # Transform test data using the same scaler
    test_X['humidity'] = scaler.transform(test_X[['humidity']])    

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