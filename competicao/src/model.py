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
train_energia.rename(columns={"Data": "data", "Hora": "hora", "Normal (kWh)": "normal", "Horário Económico (kWh)": "horario",
                               "Autoconsumo (kWh)": "autoconsumo", "Injeção na rede (kWh)": "injecao"}, inplace=True)
test_energia_X.rename(columns={"Data": "data", "Hora": "hora", "Normal (kWh)": "normal", "Horário Económico (kWh)": "horario",
                               "Autoconsumo (kWh)": "autoconsumo", "Injeção na rede (kWh)": "injecao"}, inplace=True)
print(train_energia.describe())
print(train_energia.nunique())

print(train_energia.columns)
print(train_meteo.columns)

# Prepressing energia dataset
preprocess_data(train_energia, test_energia_X)
preprocess_hora(train_energia, test_energia_X)
preprocess_normal(train_energia, test_energia_X)
preprocess_horario(train_energia, test_energia_X)
preprocess_autoconsumo(train_energia, test_energia_X)
preprocess_injecao(train_energia, test_energia_X)

# Prepressing meteo dataset
preprocess_dt(train_meteo, test_meteo_X)
preprocess_dtiso(train_meteo, test_meteo_X)
preprocess_city_name(train_meteo, test_meteo_X)
preprocess_temp(train_meteo, test_meteo_X)
preprocess_feelslike(train_meteo, test_meteo_X)
preprocess_temp_min(train_meteo, test_meteo_X)
preprocess_temp_max(train_meteo, test_meteo_X)
preprocess_pressure(train_meteo, test_meteo_X)
preprocess_sea_level(train_meteo, test_meteo_X)
preprocess_grnd_level(train_meteo, test_meteo_X)
preprocess_humidity(train_meteo, test_meteo_X)
preprocess_wind_speed(train_meteo, test_meteo_X)
preprocess_rain1h(train_meteo, test_meteo_X)
preprocess_clouds_all(train_meteo, test_meteo_X)
preprocess_weather_description(train_meteo, test_meteo_X)

print(train_energia.columns)
print(train_meteo.columns)