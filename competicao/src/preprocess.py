import pandas as pd

# PREPROCESSAMENTO ENERGIA

def preprocess_data(train, test_X):
    pass

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
    pass

def preprocess_sea_level(train, test_X):
    # all entries are null values
    train.drop(["sea_level"], inplace=True, axis=1)

def preprocess_grnd_level(train, test_X):
    # all entries are null values
    train.drop(["grnd_level"], inplace=True, axis=1)

def preprocess_humidity(train, test_X):
    pass

def preprocess_wind_speed(train, test_X):
    pass

def preprocess_rain1h(train, test_X):
    # Maybe remove? A lot of null values
    pass

def preprocess_clouds_all(train, test_X):
    pass

def preprocess_weather_description(train, test_X):
    pass

