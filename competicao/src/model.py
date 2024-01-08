import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import *
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import numpy as np
from predict import submit_prediction
from ordinal import OrdinalClassifier
from util import hold_out_validation, cross_val_score

pd.set_option('display.max_columns', None)

def get_best_model(X, y, X_valid=None, y_valid=None):
    # Reset the indices to ensure alignment
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    model_list = []

    models = [
        #('knn', KNeighborsRegressor()),
        #('mlp', MLPRegressor(random_state=random_state, hidden_layer_sizes=(100, 75), early_stopping=True,
        #                     learning_rate='invscaling', batch_size=64, max_iter=50, alpha=0,
        #                     learning_rate_init=0.0075, warm_start=True)),
        #('mlp', OrdinalClassifier(MLPClassifier(random_state=random_state, hidden_layer_sizes=(150, 100), early_stopping=False, batch_size=64, max_iter=50,
        #                     learning_rate_init=0.001, verbose=True))),
        #('dropout_mlp', OrdinalClassifier(DropoutMLP(X.shape[1]))),
        #('lstm', ModelLSTM()),
        ('rf', OrdinalClassifier(RandomForestClassifier(random_state=random_state))),
        ('gb', OrdinalClassifier(GradientBoostingClassifier(random_state=random_state))),
        #('svr', SVR()),
        ('xgb', OrdinalClassifier(XGBClassifier(random_state=random_state, enable_categorical=True, max_depth=10,
                               gamma=0.1, min_child_weight=1))),
        ('lgb', OrdinalClassifier(LGBMClassifier(random_state=random_state, verbose=-1, learning_rate=0.01,
                              lambda_l1=1, lambda_l2=1, n_estimators=1500))),
        ('cb', OrdinalClassifier(CatBoostClassifier(random_state=random_state, verbose=0)))
    ]

    for (label, model) in models:
        if (X_valid is not None) and (y_valid is not None):
            model, accuracy = hold_out_validation(model, X, y, X_valid, y_valid, label=label)
        else:
            model, accuracy = cross_val_score(model, X, y, label=label)
        model_list.append((label, model, accuracy))

    best_model_entry = max(model_list, key=lambda x: x[2])
    best_model_label = best_model_entry[0]
    best_accuracy = best_model_entry[2]
    best_model = best_model_entry[1]

    print(f"Best Model: {best_model_label} with Accuracy: {best_accuracy}")
    return best_model

if __name__== '__main__':
    # prepares data (all preprocessing included)
    train_X, train_y, test_X = data_preparation()

    # splits training set into training + validation set
    #train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.33, random_state=42)

    # gets the best model, without hyper parameter search
    best_model = get_best_model(train_X, train_y)
    # predicts the target label
    y_pred = best_model.predict(test_X)
    # rounds the target label
    y_pred_rounded = np.round(y_pred).astype(int)
    # submits the prediction
    submit_prediction(test_X, y_pred_rounded, remove_night_hours=True)