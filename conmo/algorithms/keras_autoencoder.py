from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from scipy.stats import chi2
from tensorflow.keras import layers, losses

from conmo.conf import Index, Label
from conmo.algorithms.algorithm import AnomalyDetectionThresholdBasedAlgorithm


class KerasAutoencoder(AnomalyDetectionThresholdBasedAlgorithm):

    def __init__(self, encoding_dim: int = 32, optimizer: str = 'Adam', loss_f: str = 'mse', epochs: int = 2, batch_size: int = 64, random_seed: int = 11, threshold_mode: str = 'chi2', threshold_value: Union[int, float, None] = 0.95):
        super().__init__(threshold_mode, threshold_value)
        self.encoding_dim = encoding_dim
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_seed = random_seed

        # Set random seed
        tf.random.set_seed(random_seed)

    def fit_predict(self, data_train: pd.DataFrame, data_test: pd.DataFrame, labels_train: pd.DataFrame, labels_test: pd.DataFrame) -> pd.DataFrame:
        # Build the model
        # This is our input
        input_img = keras.Input(shape=(data_train.shape[1],))
        # "encoded" is the encoded representation of the input
        encoded = layers.Dense(self.encoding_dim, activation='relu')(input_img)
        # "decoded" is the lossy reconstruction of the input
        decoded = layers.Dense(
            data_train.shape[1], activation='sigmoid')(encoded)

        # This model maps an input to its reconstruction
        autoencoder = keras.Model(input_img, decoded)

        # Compile the model
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        # TRAIN SET
        history = autoencoder.fit(data_train, data_train,
                                  epochs=self.epochs,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  validation_data=(data_test, data_test))

        # TEST SET
        # Calculate cutoff (anomaly_threshold)
        anomaly_threshold = self.find_anomaly_threshold(
            history.history['loss'], data_train.shape[1])

        # Perform classification on test set
        recons = autoencoder.predict(data_test)
        recons_err = losses.get(self.loss_f)(data_test, recons).numpy()

        # Detect anomalies
        test_error = pd.DataFrame(
            recons_err, index=data_test.index, columns=['loss'])
        test_error.loc[:, Label.ANOMALY] = test_error.loc[:,
                                                          'loss'] > anomaly_threshold

        # Generate output dataframe
        if self.labels_per_sequence(labels_test):
            # Only labels per SEQUENCE
            output = test_error.groupby(level=Index.SEQUENCE)[
                Label.ANOMALY].any()
        else:
            # Labels per TIME
            output = test_error.loc[:, Label.ANOMALY]
        output = pd.DataFrame(output, index=labels_test.index, columns=[
                              Label.ANOMALY])
        return output

    def find_anomaly_threshold(self, values: np.ndarray, n_features: int) -> float:
        if self.threshold_mode == 'chi2':
            return chi2.ppf(self.threshold_value, df=n_features)
        else:
            super().find_anomaly_threshold(values)
