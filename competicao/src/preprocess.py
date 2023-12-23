import pandas as pd
from sklearn import preprocessing

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
            return s[0:-4]
        return s
    
    # remove UTC from the end and transform to datetime
    train_meteo_df['dt_iso'] = pd.to_datetime(train_meteo_df['dt_iso'].map(remove_utc))
    test_meteo_df['dt_iso'] = pd.to_datetime(test_meteo_df['dt_iso'].map(remove_utc))
    
    # extract date and hour to different columns
    train_meteo_df['data'] = train_meteo_df['dt_iso'].dt.date
    train_meteo_df['hora'] = train_meteo_df['dt_iso'].dt.hour
    
    test_meteo_df['data'] = test_meteo_df['dt_iso'].dt.date
    test_meteo_df['hora'] = test_meteo_df['dt_iso'].dt.hour

    # make date be of type date
    train_energia_df['data'] = pd.to_datetime(train_energia_df['data']).dt.date
    test_energia_df['data'] = pd.to_datetime(test_energia_df['data']).dt.date

    print(train_meteo_df['data'].dtype)
    print(test_meteo_df['data'].dtype)
    print(train_energia_df['data'].dtype)
    print(test_energia_df['data'].dtype)

    #return the merged datasets
    # the join in the test dataset has to be outer, and we gotta predict the missing values in the meteo dataset.
    return  pd.merge(train_energia_df, train_meteo_df, how='outer'), \
            pd.merge(test_energia_df, test_meteo_df, how='inner') 

# PREPROCESSAMENTO ENERGIA

def preprocess_data(train, test_X):
    train['weekday'] = pd.to_datetime(train['data']).dt.weekday
    train['monthday'] = pd.to_datetime(train['data']).dt.day
    test_X['weekday'] = pd.to_datetime(test_X['data']).dt.weekday
    test_X['monthday'] = pd.to_datetime(test_X['data']).dt.day

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
    pass


# PREPROCESSAMENTO METEO

def preprocess_dt(train, test_X):
    pass

def preprocess_dtiso(train, test_X):
    pass

def preprocess_city_name(train, test_X):
    # all entries have the same value, so we can drop the column
    train.drop(["city_name"], inplace=True, axis=1)

def preprocess_temp(train, test_X):
    pass

def preprocess_feelslike(train, test_X):
    pass

def preprocess_temp_min(train, test_X):
    pass

def preprocess_temp_max(train, test_X):
    pass

def preprocess_pressure(train, test_X):
    train["pressure"] = preprocessing.normalize([train["pressure"]])[0]
    test_X["pressure"] = preprocessing.normalize([test_X["pressure"]])[0]


def preprocess_sea_level(train, test_X):
    # all entries are null values
    train.drop(["sea_level"], inplace=True, axis=1)

def preprocess_grnd_level(train, test_X):
    # all entries are null values
    train.drop(["grnd_level"], inplace=True, axis=1)

def preprocess_humidity(train, test_X):
    train["humidity"] = preprocessing.normalize([train["humidity"]])[0]
    test_X["humidity"] = preprocessing.normalize([test_X["humidity"]])[0]
    

def preprocess_wind_speed(train, test_X):
    pass

def preprocess_rain1h(train, test_X):
    # Maybe remove? A lot of null values
    pass

def preprocess_clouds_all(train, test_X):
    pass

def preprocess_weather_description(train, test_X):
    pass

