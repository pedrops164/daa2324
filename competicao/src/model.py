import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import *
import numpy as np
from predict import train_model, print_best_models, submit_prediction
from ensemble import EnsembleModel

pd.set_option('display.max_columns', None)

# Load input csv (energia and meteo)
df_energia_2021 = pd.read_csv('../input/energia_202109-202112.csv', encoding='latin-1', na_filter = False)
df_energia_2022 = pd.read_csv('../input/energia_202201-202212.csv', encoding='latin-1', na_filter = False)
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
train, test = merge_by_date(train_energia, train_meteo, test_energia_X, test_meteo_X)

# Prepressing energia dataset
preprocess_dates(train, test)
preprocess_hora(train, test)
preprocess_normal(train, test)
preprocess_horario(train, test)
preprocess_autoconsumo(train, test)
preprocess_injecao(train, test)

# Prepressing meteo dataset
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
train, test = preprocess_weather_description(train, test)

# fill missing values
test.to_csv("../output/intermediate_test_1.csv", index=False)
test = fill_missing_values(test)
test.to_csv("../output/intermediate_test_2.csv", index=False)

print(train.describe())
print(train.nunique())
print(train.info())
print()
print(test.describe())
print(test.nunique())
print(test.info())

train_X = train.drop(["injecao"], axis=1)
train_y = train["injecao"]

#best_model = EnsembleModel()
#best_model.fit(train_X, train_y)
#y_pred_rounded = best_model.predict(test)

best_model = print_best_models(train_X, train_y)
y_pred = best_model.predict(test)
y_pred_rounded = np.round(y_pred).astype(int)
submit_prediction(y_pred_rounded, "../output/submission.csv")