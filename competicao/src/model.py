import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import *

# Load input csv (energia and meteo)
df_energia_2021 = pd.read_csv('../input/energia_202109-202112.csv', encoding='latin-1', na_values=["", "NaN", ""])
df_energia_2022 = pd.read_csv('../input/energia_202201-202212.csv', encoding='latin-1', na_values=["", "NaN", ""])
df_meteo_2021 = pd.read_csv('../input/meteo_202109-202112.csv', encoding='latin-1', na_values=["", "NaN", ""])
df_meteo_2022 = pd.read_csv('../input/meteo_202201-202212.csv', encoding='latin-1', na_values=["", "NaN", ""])

train_energia = pd.concat([df_energia_2021, df_energia_2022])
train_meteo = pd.concat([df_meteo_2021, df_meteo_2022])

test_energia_X = pd.read_csv('../input/energia_202301-202304.csv', encoding='latin-1')
test_meteo_X = pd.read_csv('../input/meteo_202301-202304.csv', encoding='latin-1')

'''
Rename columns
'''
rename_energia(train_energia, test_energia_X)
print(train_energia.describe())
print(train_energia.nunique())

print(train_energia.columns)
print(train_meteo.columns)

train, test = merge_by_date(train_energia, train_meteo, test_energia_X, test_meteo_X)

# Prepressing energia dataset
preprocess_data(train, test)
preprocess_hora(train, test)
preprocess_normal(train, test)
preprocess_horario(train, test)
preprocess_autoconsumo(train, test)
preprocess_injecao(train, test)

# Prepressing meteo dataset
preprocess_dt(train, test)
preprocess_dtiso(train, test)
preprocess_city_name(train, test)
preprocess_temp(train, test)
preprocess_feelslike(train, test)
preprocess_temp_min(train, test)
preprocess_temp_max(train, test)
preprocess_pressure(train, test)
preprocess_sea_level(train, test)
preprocess_grnd_level(train, test)
preprocess_humidity(train, test)
preprocess_wind_speed(train, test)
preprocess_rain1h(train, test)
preprocess_clouds_all(train, test)
preprocess_weather_description(train, test)

print(train.columns)