import pandas as pd
from util import output_path, night_hours

def submit_prediction(test_X, y_pred, remove_night_hours):
    if remove_night_hours:
        # Find the indices in the test set where 'hora' is one of the hours to remove
        indices_to_remove = test_X.index[test_X['hora'].isin(night_hours)]
        ## Set predictions for these indices to 0 in y_pred_rounded
        for idx in indices_to_remove:
            y_pred[idx] = 0

    # Define the inverse of the order_mapping to translate back from numeric predictions to string labels
    inverse_order_mapping = {0: 'None', 1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
    print(y_pred.shape)
    # Map the predictions to their corresponding labels
    y_pred_labels = [inverse_order_mapping[pred] for pred in y_pred]

    # Create a DataFrame with the predictions
    submission_df = pd.DataFrame({
        'RowId': range(1, len(y_pred_labels) + 1),
        'Result': y_pred_labels
    })

    # Write the DataFrame to a CSV file
    submission_df.to_csv(output_path, index=False)