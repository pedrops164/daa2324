import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from util import random_state, cross_val_score
from ordinal import OrdinalClassifier
from preprocess import data_preparation
from predict import submit_prediction

class EnsembleModel:
    def __init__(self):
        self.models = [
            #('knn', KNeighborsRegressor()),
            #('mlp', MLPRegressor(random_state=random_state)),
            #('rf', RandomForestClassifier(random_state=random_state)),
            ('gb', OrdinalClassifier(GradientBoostingClassifier(random_state=random_state))),
            ('xgb', OrdinalClassifier(XGBClassifier(random_state=random_state, enable_categorical=True, max_depth=10,
                               gamma=0.1, min_child_weight=1))),
            ('lgb', OrdinalClassifier(LGBMClassifier(random_state=random_state, verbose=-1, learning_rate=0.01,
                              lambda_l1=1, lambda_l2=1, n_estimators=1500))),
            ('cb', OrdinalClassifier(CatBoostClassifier(random_state=random_state, verbose=0)))
        ]

    def fit(self, X_train, y_train):
        for (label, model) in self.models:
            print(f'Fitting {label}...\n')
            model.fit(X_train, y_train)

    def predict(self, X_test):
        self.result = np.zeros((X_test.shape[0]))
        for (label, model) in self.models:
            self.result = self.result + model.predict(X_test)

        self.result /= len(self.models)
        return np.round(self.result).astype(int)
    
if __name__ == '__main__':
    # only run if this is the executable
    train_X, train_y, test_X = data_preparation()

    best_model = EnsembleModel()
    best_model.fit(train_X, train_y)
    y_pred_rounded = best_model.predict(test_X)
    submit_prediction(test_X, y_pred_rounded, remove_night_hours=True)
