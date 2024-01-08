from sklearn.neural_network import MLPRegressor
from autoencoder import DAE
from util import random_state
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

class MyMLP:
    def __init__(self):
        self.mlp = MLPRegressor(random_state=random_state, hidden_layer_sizes=(100, 75), early_stopping=True,
                             learning_rate='invscaling', batch_size=64, max_iter=50, alpha=0,
                             learning_rate_init=0.0075, warm_start=True)
        self.dae = DAE()

    def fit(self, X_train, y_train, X_test=None):
        #self.dae.fit(X_train, X_test)
        #X_train = self.dae.predict(X_train)
        self.mlp.fit(X_train, y_train)

    def predict(self, X):
        #X = self.dae.predict(X)
        return self.mlp.predict(X)
    
class DropoutMLP:
    def __init__(self, input_dim):
        """
        Build a Multilayer Perceptron model with dropout layers for regularization.

        :param input_dim: int, the number of features in the input data
        :return: a compiled Keras model
        """
        self.model = Sequential([
            # First hidden layer with dropout
            Dense(500, activation='relu', input_shape=(input_dim,)),
            Dropout(0.5),  # Dropout layer with 50% dropout rate

            # Second hidden layer with dropout
            Dense(100, activation='relu'),
            Dropout(0.5),  # Another Dropout layer with 50% dropout rate
            
            # Third hidden layer with dropout
            Dense(100, activation='relu'),
            Dropout(0.5),  # Another Dropout layer with 50% dropout rate

            # Output layer
            Dense(1, activation='relu')
        ])

        # Compile the model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='mean_squared_error',  # Use this loss function for integer targets
                      metrics=['accuracy'])

        # Print the model summary
        self.model.summary()

    def fit(self, X_train, y_train, epochs=100, batch_size=64, verbose=1, validation_split=0.2):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=validation_split)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self, *args, **kwargs):
        return self.model.get_params(*args, **kwargs)