import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from util import random_state, cross_val_score

class EnsembleModel:
    def __init__(self):
        self.models = [
            #('knn', KNeighborsRegressor()),
            #('mlp', MLPRegressor(random_state=random_state)),
            ('rf', RandomForestRegressor(random_state=random_state)),
            ('gb', GradientBoostingRegressor(random_state=random_state)),
            ('xgb', XGBRegressor(random_state=random_state, enable_categorical=True)),
            ('lgb', LGBMRegressor(random_state=random_state, verbose=0, learning_rate=0.01,
                                  lambda_l1=1, lambda_l2=1, n_estimators=1000)),
            ('cb', CatBoostRegressor(random_state=random_state, verbose=0))
        ]

    def fit(self, X_train, y_train):
        for (label, model) in self.models:
            print(f'Fitting {label}...\n')
            model.fit(X_train, y_train)

    def predict(self, X_test):
        self.result = np.zeros((X_test.shape[0]))
        for (label, model) in self.models:
            self.result += model.predict(X_test)

        self.result /= len(self.models)
        return np.round(self.result).astype(int)