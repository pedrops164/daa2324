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
from util import rounded_accuracy_scorer

def grid_search(X, y):
    models = [
        ('knn', KNeighborsRegressor()),
        #('mlp', MLPRegressor(random_state=random_state, hidden_layer_sizes=(100, 75), early_stopping=True,
        #                     learning_rate='invscaling', batch_size=64, max_iter=500, alpha=0,
        #                     learning_rate_init=0.0075, warm_start=True)),
        ('rf', RandomForestClassifier(random_state=random_state)),
        #('gb', GradientBoostingClassifier(random_state=random_state)),
        #('svr', SVR()),
        #('xgb', XGBClassifier(random_state=random_state, enable_categorical=True)),
        #('lgb', LGBMClassifier(random_state=random_state, verbose=-1, learning_rate=0.01,
        #                      lambda_l1=1, lambda_l2=1, n_estimators=1500)),
        #('cb', CatBoostClassifier(random_state=random_state, verbose=0))
    ]

    hyperparameters = {
        'knn': {'n_neighbors': [2,3,4,5,6,10,14,18,25], 'weights': ['uniform','distance']},
        'mlp': {},
        'rf': {'n_estimators': [100, 200, 400], 'max_depth': [4, 8, 12, 16, 20], 'min_samples_split': [15, 20, 30, 40],
               'max_leaf_nodes': [60, 75, 100, 150]},
        'gb': {},
        'svr': {},
        'xgb': {},
        'lgb': {},
        'cb': {}
    }

    best_models = []

    for (label, model) in models:
        nfold = 3
        kfold = KFold(n_splits=nfold, shuffle=True, random_state=random_state)
        hp_space = hyperparameters[label]
        #grid = GridSearchCV(estimator=model, param_grid=hp_space, cv=kfold, scoring=rounded_accuracy_scorer)
        grid = RandomizedSearchCV(estimator=model, param_distributions=hp_space, cv=kfold, scoring=rounded_accuracy_scorer, n_iter=200, random_state=random_state, n_jobs=-1)
        model = grid.fit(X,y)

        print(f'Best Score: {grid.best_score_} | \
                Best Estimator: {grid.best_estimator_} | \
                Best Params: {grid.best_params_} | {label}\n')
        
        best_models.append(model)

    return best_models[0]
