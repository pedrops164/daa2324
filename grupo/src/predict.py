import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from tuning import *

#Tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from util import random_state, cross_val_score



def get_best_model(X, y):
    # mae_df, mse_df, r2_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Reset the indices to ensure alignment
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    models = [
        ('lin_reg', LinearRegression()),
        ('ridge', Ridge()),
        ('lasso', Lasso()),
        ('elastic_net', ElasticNet()),
        ('knn', KNeighborsRegressor()),
        #('mlp', KerasRegressor(build_fn=create_mlp_model, train=X, epochs=100, batch_size=10, verbose=0)),
        ('rf', RandomForestRegressor(random_state=random_state)),
        ('gb', GradientBoostingRegressor(random_state=random_state)),
        ('svr', SVR()),
        # ('xgb', XGBRegressor(random_state=random_state, enable_categorical=True)),
        ('lgb', LGBMRegressor(random_state=random_state, verbose=0)),
        ('cb', CatBoostRegressor(random_state=random_state, verbose=0, cat_features=['Brand_Bin']))
    ]

    model_list = []
    
    for (label, model) in models:
        model, error = cross_val_score(model, X, y, label=label)
        model_list.append((label, model, error))
    
    # we get the model with least error
    best_model_entry = min(model_list, key=lambda x : x[2])
    best_model_label, best_model, best_model_error = best_model_entry
    
    print(f"Best Model: {best_model_label} with validation error: {best_model_error}")

    return best_model, best_model_error

def create_mlp_model(train):
    # Create model
    model = Sequential()
    model.add(Dense(128, input_dim=train.shape[1], activation='relu'))  # Adjust the input_dim as per your feature count
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Output layer for regression (no activation)

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model