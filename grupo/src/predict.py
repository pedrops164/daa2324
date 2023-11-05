import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def cross_val_score(model, cv, X, y, label=''):  
    # Calculates the cross validation score of a given model with different metrics

    # mae is Mean Absolute Error
    # mse is Mean Squared Error
    # R2 is R squared 
    mae_list, mse_list, r2_list = [], [], []
    
    for train_idx, val_idx in cv.split(X, y):
        model.fit(X.iloc[train_idx], y[train_idx])
        
        val_preds = model.predict(X.iloc[val_idx])
        
        mae = mean_absolute_error(y[val_idx], val_preds)
        mse = mean_squared_error(y[val_idx], val_preds)
        r2 = r2_score(y[val_idx], val_preds)
        
        mae_list.append(mae)
        mse_list.append(mse)
        r2_list.append(r2)
    
    print(f'Val MAE: {np.mean(mae_list):.5f} ± {np.std(mae_list):.5f} | Val MSE: {np.mean(mse_list):.5f} ± {np.std(mse_list):.5f} | Val R2: {np.mean(r2_list):.5f} ± {np.std(r2_list):.5f} | {label}\n')
    
    return mae_list, mse_list, r2_list

def predict(X, y):
    mae_df, mse_df, r2_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    k = KFold(n_splits=5, shuffle=True, random_state=2023)
    models = [
        ('lin_reg', LinearRegression()),
        ('ridge', Ridge()),
        ('lasso', Lasso()),
        ('rf', RandomForestRegressor(random_state=2023)),
        ('gb', GradientBoostingRegressor(random_state=2023)),
        ('svr', SVR()),
        ('xgb', XGBRegressor(random_state=2023)),
        ('lgb', LGBMRegressor(random_state=2023)),
        ('cb', CatBoostRegressor(random_state=2023, verbose=0))
    ]

    for (label, model) in models:
        mae_df[label], mse_df[label], r2_df[label] = cross_val_score(model, k, X, y, label=label)
