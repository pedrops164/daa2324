import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.neighbors import KNeighborsRegressor
from tuning import *
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from util import rounded_accuracy_scorer, random_state

def grid_search(train_X, train_y):
    models = [
        #('knn', KNeighborsRegressor()),
        #('mlp', MLPRegressor()),
        #('rf', RandomForestClassifier(random_state=random_state)),
        #('gb', GradientBoostingClassifier(random_state=random_state)),
        #('svr', SVR()),
        ('xgb', XGBClassifier(random_state=random_state, enable_categorical=True)),
        #('lgb', LGBMClassifier(random_state=random_state, verbose=-1, learning_rate=0.01)),
        #                      lambda_l1=1, lambda_l2=1, n_estimators=1500)),
        #('cb', CatBoostClassifier(random_state=random_state, verbose=0))
    ]

    hyperparameters = {
        'knn': {'n_neighbors': [2,3,4,5,6,10,14,18,25], 'weights': ['uniform','distance']},
        'mlp': {'random_state': [random_state], 'hidden_layer_sizes': [(100,), (100, 75), (200,), (150,), (200, 150), (350, 250), (200, 200, 150), (750, 750, 500), (1500, 1000, 1000)],
                'early_stopping': [True, False], 'nesterovs_momentum': [True, False], 'beta_1': [0.9, 0.995],
                'max_iter': [200, 300], 'solver': ['sgd', 'adam'], 'alpha': [0, 0.0001, 0.0005, 0.001, 0.005, 0.01],
                'batch_size': [16, 32, 64], 'learning_rate': ['invscaling'], 'power_t': [0.5, 0.9, 0.995],
                'learning_rate_init': [0.001, 0.005, 0.01, 0.05, 0.1]},
        'rf': {'n_estimators': [100, 200, 400], 'max_depth': [4, 8, 12, 16, 20], 'min_samples_split': [15, 20, 30, 40],
               'max_leaf_nodes': [60, 75, 100, 150]},
        'gb': {},
        'svr': {},
        'xgb': {'eta': [0.1], 'max_depth':[6], 'min_child_weight':[2,3,4], 'gamma':[0.4,0.5], 'subsample':[0.8,0.85],'colsample_bytree':[0.75,0.8,0.85],
                'n_estimators':[None,1000, 5000]},
        'lgb': {'random_state': [random_state], 'learning_rate': [0.05,0.01,0.015,0.02,0.05,0.1], 
                'reg_alpha': [0, 0.0001,0.0005,0.001,0.01], 'reg_lambda': [0, 0.0001,0.0005,0.001,0.01],
                'n_estimators': [100, 50, 150, 200, 300, 500, 800], 'max_depth': [-1, 5, 10, 15, 25, 40],
                'num_leaves': [31, 15, 45, 60, 75], 'min_split_gain': [0], 'min_child_weight': [1e-3, 0.1, 1, 10, 50, 100],
                'min_child_samples': [10, 20, 30, 40, 50], 'subsample': [0.8, 0.9, 1]},
        'cb': {}
    }

    best_models = []

    for (label, model) in models:
        nfold = 3
        kfold = KFold(n_splits=nfold, shuffle=True, random_state=random_state)
        hp_space = hyperparameters[label]
        #grid = GridSearchCV(estimator=model, param_grid=hp_space, cv=kfold, scoring=rounded_accuracy_scorer)
        grid = RandomizedSearchCV(verbose=5,estimator=model, param_distributions=hp_space, cv=kfold, scoring=rounded_accuracy_scorer, n_iter=150, n_jobs=-1)
        model = grid.fit(train_X,train_y)

        print(f'Best Score: {grid.best_score_} | \
                Best Estimator: {grid.best_estimator_} | \
                Best Params: {grid.best_params_} | {label}\n')
        
        best_models.append(model)

    return best_models[0]

if __name__== '__main__':
    # prepares data (all preprocessing included)
    train_X, train_y, test_X = data_preparation()

    # splits training set into training + validation set
    #train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.33, random_state=42)

    # gets the best model, by doing grid search on the above specified models
    best_model = grid_search(train_X, train_y)
    # predicts the target label
    y_pred = best_model.predict(test_X)
    # rounds the target label
    y_pred_rounded = np.round(y_pred).astype(int)
    # submits the prediction
    submit_prediction(test_X, y_pred_rounded, remove_night_hours=True)
