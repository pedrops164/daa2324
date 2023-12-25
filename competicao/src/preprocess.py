import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

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
    train_meteo_df['hora'] = train_meteo_df['dt_iso'].dt.hour
    
    test_meteo_df['data'] = test_meteo_df['dt_iso'].dt.date
    test_meteo_df['ano'] = test_meteo_df['dt_iso'].dt.year
    test_meteo_df['mes'] = test_meteo_df['dt_iso'].dt.month
    test_meteo_df['hora'] = test_meteo_df['dt_iso'].dt.hour

    # create year month and day columns for energia dataset
    train_energia_datetime = pd.to_datetime(train_energia_df['data'])
    train_energia_df['data'] = train_energia_datetime.dt.date
    train_energia_df['ano'] = train_energia_datetime.dt.year
    train_energia_df['mes'] = train_energia_datetime.dt.month

    test_energia_datetime = pd.to_datetime(test_energia_df['data'])
    test_energia_df['data'] = test_energia_datetime.dt.date
    test_energia_df['ano'] = test_energia_datetime.dt.year
    test_energia_df['mes'] = test_energia_datetime.dt.month

    print(train_meteo_df['data'].dtype)
    print(test_meteo_df['data'].dtype)
    print(train_energia_df['data'].dtype)
    print(test_energia_df['data'].dtype)

    #return the merged datasets
    # the join in the test dataset has to be outer, and we gotta predict the missing values in the meteo dataset.
    return  pd.merge(train_energia_df, train_meteo_df, how='inner'), \
            pd.merge(test_energia_df, test_meteo_df, how='outer') 

# PREPROCESSAMENTO ENERGIA

# preprocessing of dates
def preprocess_dates(train, test_X):
    train['weekday'] = pd.to_datetime(train['data']).dt.weekday
    train['monthday'] = pd.to_datetime(train['data']).dt.day
    test_X['weekday'] = pd.to_datetime(test_X['data']).dt.weekday
    test_X['monthday'] = pd.to_datetime(test_X['data']).dt.day

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
    order_mapping = {'None': 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Very High': 5}
    train['injecao'] = train['injecao'].map(order_mapping)
    #le = LabelEncoder()
    #train['injecao'] = le.fit_transform(train['injecao'])
    pass


# PREPROCESSAMENTO METEO

def preprocess_dt(train, test_X):
    pass

def preprocess_dtiso(train, test_X):
    pass

def preprocess_city_name(train, test_X):
    # all entries have the same value, so we can drop the column
    train.drop(["city_name"], inplace=True, axis=1)
    test_X.drop(["city_name"], inplace=True, axis=1)

def preprocess_temp(train, test_X):
    pass

def preprocess_feelslike(train, test_X):
    pass

def preprocess_temp_min(train, test_X):
    pass

def preprocess_temp_max(train, test_X):
    pass

def preprocess_pressure(train, test_X):
    # Fill missing values with median
    pressure_median = train['pressure'].median()
    train['pressure'].fillna(pressure_median, inplace=True)
    test_X['pressure'].fillna(pressure_median, inplace=True)

    # normalize
    train["pressure"] = preprocessing.normalize([train["pressure"]])[0]
    test_X["pressure"] = preprocessing.normalize([test_X["pressure"]])[0]


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
    humidity_mean = train['humidity'].mean()
    train['humidity'].fillna(humidity_mean, inplace=True)
    test_X['humidity'].fillna(humidity_mean, inplace=True)

    # normalization
    train["humidity"] = preprocessing.normalize([train["humidity"]])[0]
    test_X["humidity"] = preprocessing.normalize([test_X["humidity"]])[0]
    

def preprocess_wind_speed(train, test_X):
    pass

def preprocess_rain1h(train, test_X):
    # Maybe remove? A lot of null values
    train.drop(["rain_1h"], inplace=True, axis=1)
    test_X.drop(["rain_1h"], inplace=True, axis=1)

def preprocess_clouds_all(train, test_X):
    pass

def preprocess_weather_description(train, test_X):
    # since the weather_description column only has 8 possible values, one hot encode it
    ohe_train = pd.get_dummies(train['weather_description'], prefix='wd')
    train = train.join(ohe_train)
    train.drop("weather_description", axis=1, inplace=True)

    ohe_test = pd.get_dummies(test_X['weather_description'], prefix='wd')
    test_X = test_X.join(ohe_test)
    test_X.drop("weather_description", axis=1, inplace=True)

    return train, test_X