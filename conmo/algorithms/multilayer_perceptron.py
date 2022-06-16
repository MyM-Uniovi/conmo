import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from conmo.conf import Label, RandomSeed
from conmo.algorithms.algorithm import PretrainedAlgorithm


class PretrainedMultilayerPerceptron(PretrainedAlgorithm):

    def __init__(self, pretrained: bool, input_len: int, random_seed: int = None, path: str = None) -> None:
        super().__init__(pretrained, path)
        self.input_len = input_len
        if not self.pretrained:
            if random_seed != None:
                self.random_seed = random_seed
            else:
                self.random_seed = RandomSeed.RANDOM_SEED

    def fit_predict(self, data_train: pd.DataFrame, data_test: pd.DataFrame, labels_train: pd.DataFrame, labels_test: pd.DataFrame) -> pd.DataFrame:
        self.model = None
        if not self.pretrained:
            # Set TensorFlow random seed
            tf.random.set_seed(self.random_seed)

            # Create new Multilayer Perceptron
            self.model = self.build_mlp(self)

            # Compile model
            self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])

            # Train model with only train data
            self.model.fit(data_train.to_numpy(), labels_train.to_numpy(
            ), epochs=100, batch_size=32, validation_split=0.2, verbose=1)
        else:
            # If there is a pretrained model saved the is no reason to train
            # Load weights from disk
            self.load_weights()

        # Predict over test set
        pred = self.model.predict(data_test.to_numpy())

        # Generate output dataframe
        return pd.DataFrame(pred, index=labels_test.index, columns=Label.BATTERIES_DEG_TYPES)

    def load_weights(self) -> None:
        self.model = tf.keras.models.load_model(self.path, compile=False)

    def build_mlp(self) -> Sequential:
        """
        Auxiliary method for building the multilayer percetron.

        Returns
        -------
        model: tf.keras.Model
            Keras model built.
        """
        model = Sequential()
        model.add(Dense(64, input_dim=self.input_len, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(3, activation='sigmoid'))
        return model
