from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, log_loss, confusion_matrix
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score
import pandas as pd

random_state = 2023

def cross_val_score(model, X, y, label=''):  
    # Calculates the cross validation score of a given model with different metrics

    cv = KFold(n_splits=3, shuffle=True, random_state=random_state)

    # mae is Mean Absolute Error
    mae_list, acc_val_list, acc_train_list = [], [], []
    # Initialize lists to store binary confusion matrices for each class
    binary_conf_matrices = {class_label: [] for class_label in np.unique(y)}

    for train_idx, val_idx in cv.split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        
        val_preds = model.predict(X.iloc[val_idx])
        val_preds_rounded = np.round(val_preds).astype(int)

        train_preds = model.predict(X.iloc[train_idx])
        train_preds_rounded = np.round(train_preds).astype(int)

        for class_label in binary_conf_matrices.keys():
            # Treat the current class_label as the positive class (1)
            true_binary = (y.iloc[val_idx] == class_label).astype(int)
            pred_binary = (val_preds_rounded == class_label).astype(int)

            # Calculate the binary confusion matrix and append it
            binary_conf_matrix = confusion_matrix(true_binary, pred_binary, labels=[1, 0])
            binary_conf_matrices[class_label].append(binary_conf_matrix)

        mae_val = mean_absolute_error(y.iloc[val_idx], val_preds)
        acc_val = accuracy_score(y.iloc[val_idx], val_preds_rounded)
        acc_train = accuracy_score(y.iloc[train_idx], train_preds_rounded)
        
        mae_list.append(mae_val)
        acc_val_list.append(acc_val)
        acc_train_list.append(acc_train)
    
    # Aggregate and average the binary confusion matrices
    avg_binary_conf_matrices = {}
    for class_label, matrices in binary_conf_matrices.items():
        avg_matrix = np.mean(matrices, axis=0)
        avg_binary_conf_matrices[class_label] = avg_matrix
    
    print(f'Val Accuracy: {np.mean(acc_val_list):.5f} | Training Accuracy: {np.mean(acc_train_list):.5f} | {label}\n')
    # Print the average binary confusion matrices
    #print(f'{label} - Average Binary Confusion Matrices:')
    #for class_label, avg_matrix in avg_binary_conf_matrices.items():
    #    print(f'Class {class_label}:')
    #    print(avg_matrix)
    #    print()  # Blank line for readability
    #print(f'Val MAE: {np.mean(mae_list):.5f} ± {np.std(mae_list):.5f} | Accuracy: {np.mean(acc_list):.5f} | {label}\n')
    #print(f'Val MAE: {np.mean(mae_list):.5f} ± {np.std(mae_list):.5f} | Val MSE: {np.mean(mse_list):.5f} ± {np.std(mse_list):.5f} | Val R2: {np.mean(r2_list):.5f} ± {np.std(r2_list):.5f} | {label}\n')
    
    # train model with all of the data and return it
    model.fit(X, y)

    # returns the model and the average accuracy from the cross validation
    return model, np.mean(acc_val_list), np.mean(mae_list)

def rounded_accuracy(y_true, y_pred):
    y_pred_rounded = np.round(y_pred)  # Round the predictions
    return accuracy_score(y_true, y_pred_rounded)  # Calculate accuracy with rounded predictions

# Create a custom scorer that you can pass to GridSearchCV
rounded_accuracy_scorer = make_scorer(rounded_accuracy)