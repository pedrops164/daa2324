from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, log_loss, confusion_matrix
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

random_state = 2023
output_path = '../output/submission.csv'
night_hours = [0, 1, 2, 3, 4, 5, 20, 21, 22, 23]

def plot_learning_curve(train_curve, label, output_dir):
    plt.figure()
    plt.figure(figsize=(10, 6))
    plt.plot(train_curve)
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)

    # Constructing the file name
    file_name = f"{label}_learning_curve.png"
    file_path = os.path.join(output_dir, file_name)

    # Saving the plot to a file
    plt.savefig(file_path)
    print(f"Plot saved to {file_path}")
    plt.close()  # Close the plot to free up memory

def cross_val_score(model, X, y, label='', print_conf_matrix=False):  
    # Calculates the cross validation score of a given model with different metrics

    cv = KFold(n_splits=2, shuffle=True, random_state=random_state)

    # mae is Mean Absolute Error
    acc_val_list, acc_train_list = [], []
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

        acc_val = accuracy_score(y.iloc[val_idx], val_preds_rounded)
        acc_train = accuracy_score(y.iloc[train_idx], train_preds_rounded)
        
        acc_val_list.append(acc_val)
        acc_train_list.append(acc_train)
    
    # Aggregate and average the binary confusion matrices
    avg_binary_conf_matrices = {}
    for class_label, matrices in binary_conf_matrices.items():
        avg_matrix = np.mean(matrices, axis=0)
        avg_binary_conf_matrices[class_label] = avg_matrix
    
    if print_conf_matrix:
        # Print the average binary confusion matrices
        print(f'{label} - Average Binary Confusion Matrices:')
        for class_label, avg_matrix in avg_binary_conf_matrices.items():
            print(f'Class {class_label}:')
            print(avg_matrix)
            print()  # Blank line for readability
    
    print(f'Val Accuracy: {np.mean(acc_val_list):.5f} | Training Accuracy: {np.mean(acc_train_list):.5f} | {label}\n')
    # now we train the model with all of the data and return it
    model.fit(X, y)

    # returns the model and the average accuracy from the cross validation
    return model, np.mean(acc_val_list)

def hold_out_validation(model, X_train, y_train, X_validation, y_validation, label='', print_conf_matrix=False):  
    # Initialize dictionary to store binary confusion matrices for each class

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the training data
    train_preds = model.predict(X_train)
    train_preds_rounded = np.round(train_preds).astype(int)
    
    # Make predictions on the validation data
    val_preds = model.predict(X_validation)
    val_preds_rounded = np.round(val_preds).astype(int)

    if print_conf_matrix:
        binary_conf_matrices = {class_label: [] for class_label in np.unique(y_train)}
        #Calculate the binary confusion matrix for each class
        for class_label in binary_conf_matrices.keys():
            true_binary = (y_validation == class_label).astype(int)
            pred_binary = (val_preds_rounded == class_label).astype(int)
            binary_conf_matrix = confusion_matrix(true_binary, pred_binary, labels=[1, 0])
            binary_conf_matrices[class_label] = binary_conf_matrix

    # Calculate Mean Absolute Error and Accuracy for validation data
    acc_val = accuracy_score(y_validation, val_preds_rounded)
    acc_train = accuracy_score(y_train, train_preds_rounded)

    print(f'Val Accuracy: {acc_val:.5f} | Training Accuracy: {acc_train:.5f} | {label}\n')

    # Return the model and its accuracy
    return model, acc_val

def rounded_accuracy(y_true, y_pred):
    y_pred_rounded = np.round(y_pred)  # Round the predictions
    return accuracy_score(y_true, y_pred_rounded)  # Calculate accuracy with rounded predictions

# Create a custom scorer that we can pass to GridSearchCV
rounded_accuracy_scorer = make_scorer(rounded_accuracy)