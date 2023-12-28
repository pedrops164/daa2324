#LSTM model (Long Short-Term Memory)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
import math

# This function keeps the initial learning rate for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * math.exp(-0.1)

class ModelLSTM:
    def __init__(self, seq_length=5):
        self.seq_length = seq_length
        pass

    def fit(self, X, y):
        self.model = Sequential()
        self.model.add(LSTM(100, activation='relu', input_shape=(self.seq_length, X.shape[1])))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(75, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer=Adam(learning_rate=0.0001),
                           loss='mean_squared_error')

        # Convert X and y to numpy arrays for indexing
        X = X.to_numpy()
        y = y.to_numpy()
        # create temporal sequences
        xs = []
        ys = []
        for i in range(len(X)-self.seq_length-1):
            x_block = X[i:(i+self.seq_length)]
            y_block = y[i+self.seq_length]  # target_index é o índice da coluna alvo
            xs.append(x_block)
            ys.append(y_block)
    
        # fit
        self.model.fit(np.array(xs), np.array(ys), epochs=50, batch_size=64,
                       callbacks=[EarlyStopping(patience=5, restore_best_weights=True),
                                  LearningRateScheduler(scheduler)],
                       validation_split=0.1)

    def predict(self, X_test):
        # Convert X_test to numpy array if it isn't one already
        X_test = X_test.to_numpy() if not isinstance(X_test, np.ndarray) else X_test

        # Create temporal sequences from the test data
        xs_test = []
        for i in range(len(X_test) - self.seq_length + 1):  # Adjusted to ensure we use all available data
            x_block = X_test[i:(i + self.seq_length)]
            xs_test.append(x_block)

        # Make predictions using the model
        xs_test = np.array(xs_test)
        y_pred = self.model.predict(xs_test)
        y_pred = y_pred.ravel() # change shape from (n_examples, 1) to (n_examples,). Or 2D to 1D
        return np.concatenate((np.zeros(self.seq_length - 1), y_pred))
    