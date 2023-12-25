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



def print_best_models(X, y):
    mae_df, mse_df, r2_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Reset the indices to ensure alignment
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    model_list = []

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
        ('xgb', XGBRegressor(random_state=random_state, enable_categorical=True)),
        ('lgb', LGBMRegressor(random_state=random_state, verbose=0)),
        ('cb', CatBoostRegressor(random_state=random_state, verbose=0))
    ]

    for (label, model) in models:
        model, accuracy = cross_val_score(model, X, y, label=label)
        
        model_list.append((label, model, accuracy))

    best_model_entry = max(model_list, key=lambda x: x[2])
    best_model_label = best_model_entry[0]
    best_accuracy = best_model_entry[2]
    best_model = best_model_entry[1]

    print(f"Best Model: {best_model_label} with Accuracy: {best_accuracy}")
    return best_model

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

def train_model(X, y):
    '''
    We know the best models are
    - random forest regressor
    - gradient boosting regressor
    - xgb regressor
    - lgbm regressor
    - cat boost regressor

    So we are going to tune the hyperparameters of each of these models, and then ensemble them
    '''
    cb_best = bestparams_cb(X, y)
    lgbm_best = bestparams_lgbm(X, y)
    gr_best = best_params_gradientboost(X, y)

    lgbm_reg = LGBMRegressor(learning_rate=lgbm_best['learning_rate'],
                            min_child_samples=int(lgbm_best['min_child_samples']),
                            n_estimators=int(lgbm_best['n_estimators']),
                            num_leaves=int(lgbm_best['num_leaves']),
                            objective='mae',
                            random_state=random_state)

    gr_reg = GradientBoostingRegressor(
                                n_estimators=int(gr_best['n_estimators']), 
                                max_depth=int(gr_best['max_depth']),
                                learning_rate=gr_best['learning_rate'],
                                loss='absolute_error',
                                random_state=random_state)

    cb_reg = CatBoostRegressor(learning_rate=cb_best['learning_rate'], 
                                iterations=cb_best['iterations'],
                                l2_leaf_reg=cb_best['l2_leaf_reg'],
                                depth=cb_best['depth'],
                                bootstrap_type='Bayesian',
                                objective='MAE',
                                silent=True,
                                random_state=random_state)
    
    models = [lgbm_reg, gr_reg, cb_reg]
    models = [
        ('lgbm', lgbm_reg),
        ('gradient boost', gr_reg),
        ('cat boost', cb_reg),
    ]
    best_scores = []
    for (label, model) in models:
        best_scores.append(cross_val_score(model, X, y, label=label))

    print(best_scores)

def submit_prediction(y_pred, output_path):
    # Define the inverse of the order_mapping to translate back from numeric predictions to string labels
    inverse_order_mapping = {1: 'None', 2: 'Low', 3: 'Medium', 4: 'High', 5: 'Very High'}

    # Map the predictions to their corresponding labels
    y_pred_labels = [inverse_order_mapping[pred] for pred in y_pred]

    # Create a DataFrame with the predictions
    submission_df = pd.DataFrame({
        'RowId': range(1, len(y_pred_labels) + 1),
        'Result': y_pred_labels
    })

    # Write the DataFrame to a CSV file
    submission_df.to_csv(output_path, index=False)