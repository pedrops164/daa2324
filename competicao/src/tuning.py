
from hyperopt import hp
from hyperopt import fmin, tpe, Trials 
import numpy as np
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from util import random_state, cross_val_score
from sklearn.ensemble import RandomForestRegressor
'''
Hyperparameter tuning
'''

def best_params(objective_func, search_space):
    trials = Trials()

    best = fmin(fn=objective_func,
                space=search_space,
                algo=tpe.suggest,
                max_evals=50,
                trials=trials,
                rstate=np.random.default_rng(seed=30))

    print('Beat Parameters:', best)
    return best 

def bestparams_cb(X, y):
    # best parameters for cat boost regressor
    cb_search_space = {'learning_rate': hp.uniform('learning_rate', 0.01, 0.05),
                    'iterations': hp.randint('iterations',100,1000),
                    'l2_leaf_reg': hp.randint('l2_leaf_reg',1,10),
                    'depth': hp.randint('depth',4,10),
                    'bootstrap_type' : hp.choice('bootstrap_type', ['Bayesian', 'Bernoulli'])}

    def objective_func(search_space):
        cb_reg = CatBoostRegressor(learning_rate=search_space['learning_rate'], 
                                iterations=search_space['iterations'],
                                l2_leaf_reg=search_space['l2_leaf_reg'],
                                depth=search_space['depth'],
                                bootstrap_type=search_space['bootstrap_type'],
                                objective='MAE',
                                silent=True,
                                random_state=random_state)
        model, acc, mae = cross_val_score(cb_reg, X, y)
        return mae
    return best_params(objective_func, cb_search_space)

def bestparams_lgbm(X, y):
    # best parameters for lightgbm
    lgbm_search_space = {
        'num_leaves': hp.quniform('num_leaves', 10, 100, 10),
        'min_child_samples': hp.quniform('min_child_samples', 20, 200, 10),
        'learning_rate': hp.uniform('learning_rate', 0.1, 0.5),
        'n_estimators': hp.quniform('n_estimators', 100, 300, 50)
    }

    def objective_func(search_space):
        lgbm_reg = LGBMRegressor(num_leaves=int(search_space['num_leaves']),
                                min_child_samples=int(search_space['min_child_samples']),
                                learning_rate=search_space['learning_rate'],
                                n_estimators=int(search_space['n_estimators']),
                                random_state=random_state)
        model, acc, mae = cross_val_score(lgbm_reg, X, y)
        return mae
    return best_params(objective_func, lgbm_search_space)

def best_params_gradientboost(X, y):
    # best parameters for gradient boost
    gr_search_space = {
        'n_estimators': hp.quniform('n_estimators', 100, 300, 50),
        'max_depth': hp.quniform('max_depth', 1, 10, 2),
        'learning_rate': hp.uniform('learning_rate', 0.1, 0.5)
    }

    def objective_func(search_space):
        gr_reg = GradientBoostingRegressor(n_estimators=int(search_space['n_estimators']), 
                                max_depth=int(search_space['max_depth']),
                                learning_rate=search_space['learning_rate'],
                                random_state=random_state)
        model, acc, mae = cross_val_score(gr_reg, X, y)
        return mae

    return best_params(objective_func, gr_search_space)

def best_params_random_forest(X, y):
    rf_search_space = {
        'n_estimators': hp.quniform('n_estimators', 100, 500, 50),
        'max_depth': hp.quniform('max_depth', 10, 15, 20),
        'min_samples_split': hp.quniform('min_samples_split', 2, 5, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 2, 6, 1),
        'max_features': hp.uniform('max_features', 0.1, 1.0)
    }

    def objective_func(search_space):
        rf_regressor = RandomForestRegressor(
            n_estimators=int(search_space['n_estimators']),
            max_depth=int(search_space['max_depth']),
            min_samples_split=int(round(search_space['min_samples_split'])),
            min_samples_leaf=int(search_space['min_samples_leaf']),
            max_features=search_space['max_features'],
            random_state=2023
        )

        model, acc, mae = cross_val_score(rf_regressor, X, y)
        return mae

    return best_params(objective_func, rf_search_space)

