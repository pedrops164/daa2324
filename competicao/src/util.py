from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, log_loss
import numpy as np

random_state = 2023

def cross_val_score(model, X, y, label=''):  
    # Calculates the cross validation score of a given model with different metrics

    cv = KFold(n_splits=3, shuffle=True, random_state=random_state)

    # mae is Mean Absolute Error
    mae_list, acc_val_list, acc_train_list = [], [], []
    
    for train_idx, val_idx in cv.split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        
        val_preds = model.predict(X.iloc[val_idx])
        val_preds_rounded = np.round(val_preds).astype(int)

        train_preds = model.predict(X.iloc[train_idx])
        train_preds_rounded = np.round(train_preds).astype(int)

        mae_val = mean_absolute_error(y.iloc[val_idx], val_preds)
        acc_val = accuracy_score(y.iloc[val_idx], val_preds_rounded)
        acc_train = accuracy_score(y.iloc[train_idx], train_preds_rounded)
        
        mae_list.append(mae_val)
        acc_val_list.append(acc_val)
        acc_train_list.append(acc_train)
    
    print(f'Val Accuracy: {np.mean(acc_val_list):.5f} | Training Accuracy: {np.mean(acc_train_list):.5f} | {label}\n')
    #print(f'Val MAE: {np.mean(mae_list):.5f} ± {np.std(mae_list):.5f} | Accuracy: {np.mean(acc_list):.5f} | {label}\n')
    #print(f'Val MAE: {np.mean(mae_list):.5f} ± {np.std(mae_list):.5f} | Val MSE: {np.mean(mse_list):.5f} ± {np.std(mse_list):.5f} | Val R2: {np.mean(r2_list):.5f} ± {np.std(r2_list):.5f} | {label}\n')
    
    # train model with all of the data and return it
    model.fit(X, y)

    # returns the model and the average accuracy from the cross validation
    return model, np.mean(acc_val_list), np.mean(mae_list)