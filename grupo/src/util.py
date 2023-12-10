from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

random_state = 2023

def cross_val_score(model, X, y, label=''):  
    # Calculates the cross validation score of a given model with different metrics

    cv = KFold(n_splits=3, shuffle=True, random_state=random_state)

    # mae is Mean Absolute Error
    # mse is Mean Squared Error
    # R2 is R squared 
    mae_list, mse_list, r2_list = [], [], []
    
    for train_idx, val_idx in cv.split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        
        val_preds = model.predict(X.iloc[val_idx])
        
        mae = mean_absolute_error(y.iloc[val_idx], val_preds)
        #mse = mean_squared_error(y[val_idx], val_preds)
        #r2 = r2_score(y[val_idx], val_preds)
        
        mae_list.append(mae)
        #mse_list.append(mse)
        #r2_list.append(r2)
    
    print(f'Val MAE: {np.mean(mae_list):.5f} ± {np.std(mae_list):.5f} | {label}\n')
    #print(f'Val MAE: {np.mean(mae_list):.5f} ± {np.std(mae_list):.5f} | Val MSE: {np.mean(mse_list):.5f} ± {np.std(mse_list):.5f} | Val R2: {np.mean(r2_list):.5f} ± {np.std(r2_list):.5f} | {label}\n')
    
    return np.mean(mae_list)