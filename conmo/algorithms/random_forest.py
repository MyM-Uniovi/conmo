import pandas as pd
from joblib import load
from sklearn.ensemble import RandomForestRegressor

from conmo.conf import Index, Label, RandomSeed
from conmo.algorithms.algorithm import PretrainedAlgorithm


class PretrainedRandomForest(PretrainedAlgorithm):

    def __init__(self, pretrained: bool, max_depth: int = None, random_state: int = None, n_estimators: int = None, path: str = None) -> None:
        super().__init__(pretrained, path)
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        if not self.pretrained:
            if random_state != None:
                self.random_seed = random_state
            else:
                self.random_seed = RandomSeed.RANDOM_SEED

    def fit_predict(self, data_train: pd.DataFrame, data_test: pd.DataFrame, labels_train: pd.DataFrame, labels_test: pd.DataFrame) -> pd.DataFrame:
        self.model = None
        if not self.pretrained:
            # Create new Random Forest Regressor
            self.model = RandomForestRegressor(
                max_depth=self.max_depth, random_state=self.random_state, n_estimators=self.n_estimators)

            # Train model with only train data
            self.model.fit(data_train.to_numpy(), labels_train.to_numpy())
        else:
            # If there is a pretrained model saved the is no reason to train
            # Load weights from disk
            self.load_weights()

        # Predict over test set
        pred = self.model.predict(data_test.to_numpy())

        # Generate output dataframe
        return pd.DataFrame(pred, index=labels_test.index, columns=Label.BATTERIES_DEG_TYPES)

    def load_weights(self):
        self.model = load(self.path)
