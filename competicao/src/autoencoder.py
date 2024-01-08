from keras.layers import Input, Dense, BatchNormalization, LeakyReLU
from keras.models import Model
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # for plotting model loss

def add_swap_noise(df, swap_prob=0.15):
    # Create a copy of the dataframe to hold the noisy version
    noisy_df = df.copy()
    
    # Determine the number of features to swap for each row
    n_cols = df.shape[1]
    
    for col in range(n_cols):
        # Create a boolean array representing if an element in the column should be swapped
        swap = np.random.rand(len(df)) < swap_prob
        
        # Create a random permutation of row indices
        random_rows = np.random.permutation(len(df))
        
        # Swap the selected elements with the corresponding elements in the randomly permuted row
        # Using .values to ensure direct indexing without considering pandas index
        original_values = df.iloc[:, col].values
        noisy_df.iloc[swap, col] = original_values[random_rows][swap]

    return noisy_df

class DAE:
    def __init__(self):
        pass

    def fit(self, train, test):
        df = pd.concat([train,test])
        # Add artificial noise
        #noise_factor = 0.5
        #df_noisy = df + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=df.shape) 
        #df_noisy = np.clip(df_noisy, 0., 1.)

        # Apply swap noise
        df_noisy = add_swap_noise(df, swap_prob=0.15)

        n_neurons_hidden = 150

        #--- Define Shapes
        n_inputs=df.shape[1] # number of input neurons = the number of features X_train

        #--- Input Layer 
        visible = Input(shape=(n_inputs,), name='Input-Layer') # Specify input shape

        #--- Encoder Layer
        e = Dense(units=n_inputs, name='Encoder-Layer')(visible)
        e = BatchNormalization(name='Encoder-Layer-Normalization')(e)
        e = LeakyReLU(name='Encoder-Layer-Activation')(e)
        
        #--- Middle Layer
        middle1 = Dense(units=n_neurons_hidden, activation='relu', activity_regularizer=keras.regularizers.L1(0.0001), name='Middle-Hidden-Layer1')(e)
        middle2 = Dense(units=n_neurons_hidden, activation='relu', activity_regularizer=keras.regularizers.L1(0.0001), name='Middle-Hidden-Layer2')(middle1)
        middle3 = Dense(units=n_neurons_hidden, activation='relu', activity_regularizer=keras.regularizers.L1(0.0001), name='Middle-Hidden-Layer3')(middle2)
        
        #--- Decoder Layer
        d = Dense(units=n_inputs, name='Decoder-Layer')(middle3)
        d = BatchNormalization(name='Decoder-Layer-Normalization')(d)
        d = LeakyReLU(name='Decoder-Layer-Activation')(d)
        
        #--- Output layer
        output = Dense(units=n_inputs, activation='sigmoid', name='Output-Layer')(d)
        
        # Define denoising autoencoder model
        self.model = Model(inputs=visible, outputs=output, name='Denoising-Autoencoder-Model')
        
        # Compile denoising autoencoder model
        self.model.compile(optimizer='adam', loss='mse')
        # Print model summary
        print(self.model.summary())

        # Fit the Denoising autoencoder model to reconstruct original data
        history = self.model.fit(df_noisy, df, epochs=20, batch_size=32, verbose=1, validation_split=0.2)

        # Plot a loss chart
        fig, ax = plt.subplots(figsize=(16,9), dpi=300)
        plt.title(label='Model Loss by Epoch', loc='center')

        ax.plot(history.history['loss'], label='Training Data', color='black')
        ax.plot(history.history['val_loss'], label='Test Data', color='red')
        ax.set(xlabel='Epoch', ylabel='Loss')
        plt.xticks(ticks=np.arange(len(history.history['loss'])), labels=np.arange(1, len(history.history['loss'])+1))
        plt.legend()
        # Save the plot to a file
        plt.savefig('../output/dae_loss_curve.png', format='png')
        plt.close(fig)

    def predict(self, X):
        X_denoised = self.model.predict(X)
        return X_denoised