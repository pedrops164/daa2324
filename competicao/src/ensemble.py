import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from util import random_state, cross_val_score

class EnsembleModel:
    def __init__(self):
        self.models = [
            #('knn', KNeighborsRegressor()),
            #('mlp', MLPRegressor(random_state=random_state)),
            ('rf', RandomForestClassifier(random_state=random_state)),
            ('gb', GradientBoostingClassifier(random_state=random_state)),
            ('xgb', XGBClassifier(random_state=random_state, enable_categorical=True)),
            ('lgb', LGBMClassifier(random_state=random_state, verbose=-1, learning_rate=0.01,
                              lambda_l1=1, lambda_l2=1, n_estimators=1500)),
            #('cb', CatBoostClassifier(random_state=random_state, verbose=0))
        ]

    def fit(self, X_train, y_train):
        for (label, model) in self.models:
            print(f'Fitting {label}...\n')
            model.fit(X_train, y_train)

    def predict(self, X_test):
        self.result = np.zeros((X_test.shape[0]))
        for (label, model) in self.models:
            print(self.result.shape)
            self.result = self.result + model.predict(X_test)
        print(self.result.shape)

        self.result /= len(self.models)
        return np.round(self.result).astype(int)