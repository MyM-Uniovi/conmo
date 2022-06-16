from abc import ABC, abstractmethod
from os import path

import numpy as np
import pandas as pd

from conmo.conf import File, Index


class Algorithm(ABC):

    @abstractmethod
    def fit_predict(self, data_train: pd.DataFrame, data_test: pd.DataFrame, labels_train: pd.DataFrame, labels_test: pd.DataFrame) -> pd.DataFrame:
        """
        Trains the model with train data and then performs predictions with the trained algorithm over the test data.

        Parameters
        ----------
        data_train: Pandas Dataframe
            Train data.
        data_test: Pandas Dataframe
            Test data.
        labels_train: Pandas Dataframe
            Train labels.
        labels_test: Pandas Dataframe
            Test labels.

        Returns
        -------
        Pandas Dataframe
            Results of the predictions made on the test set.
        """
        pass

    def execute(self, idx: int, in_dir: str, out_dir: str) -> str:
        """
        Performs a complete execution of the algorithm, loading input data, 
        performing a run through the folds and saving the results.

        Parameters
        ----------
        idx: int
            Index of the algorithm in the Experiment. Userful in case you want to experiment with several algorithms.
        in_dir: str
            Intermediate directory where the input data to the algorithm is stored.
        out_dir: str
            Intermediate directory where the output data (predictios of the algorithm) will be stored.

        Returns
        -------
        str
            Name of the output directory.

        """
        self.show_start_message()

        # Load input data
        data, labels = self.load_input(in_dir)
        folds = data.index.get_level_values(Index.FOLD).unique()

        # Train/test over all folds
        results = []
        for fold in folds:
            print("Fold {:02}/{:02}".format(fold, len(folds)))
            data_train = data.loc[fold, Index.SET_TRAIN]
            data_test = data.loc[fold, Index.SET_TEST]
            labels_train = labels.loc[fold, Index.SET_TRAIN]
            labels_test = labels.loc[fold, Index.SET_TEST]

            results.append(self.fit_predict(
                data_train, data_test, labels_train, labels_test))

        # Save results
        results = pd.concat(results, keys=folds, names=[
                            Index.FOLD].append(labels_test.index.names))
        return self.save_output(results, out_dir, idx)

    def labels_per_sequence(self, labels: pd.DataFrame) -> bool:
        """
        Use only with time series datasets.
        Checks if the labels file of the chosen dataset has an index format with sequences only or sequences and time.
        *This method in future updates will be changed to a specific class for time series.*

        Parameters
        ----------
        labels: Pandas Dataframe
            Labels file of the dataset.
        
        Returns
        -------
        bool
            True if the labels contains 1 level of index with sequence or False if the labels file contains 2 leves with sequence
            and time.

        Raises
        ------
        RuntimeError
            If the number of index levels is invalid.
        """
        if labels.index.nlevels == 1 and labels.index.names[0] == Index.SEQUENCE:
            return True
        elif labels.index.nlevels == 2 and labels.index.names[0] == Index.SEQUENCE and labels.index.names[1] == Index.TIME:
            return False
        else:
            raise RuntimeError("Invalid number of levels for labels.")

    def show_start_message(self):
        """
        Simple method to print on the terminal the name of the algorithm to be executed.
        """
        print("\n+++ Algorithm {} +++".format(self.__class__.__name__))

    def load_input(self, in_dir: str) -> (pd.DataFrame, pd.DataFrame):
        """
        Read parquet data and labels files of the chosen dataset.

        Parameters
        ----------
        in_dir: str
            Input directory where the files are located.

        Returns
        -------
        data: Pandas Dataframe
            Loaded data file.
        labels: Pandas Dataframe
            Loaded labels file.
        """
        data = pd.read_parquet(path.join(in_dir, File.DATA))
        labels = pd.read_parquet(path.join(in_dir, File.LABELS))
        return data, labels

    def save_output(self, results: pd.DataFrame, out_dir: str, idx: int) -> str:
        """
        Save algorithms output to parquet format.

        Parameters
        ----------
        results: Pandas Dataframe
            Dataframe with the results of the execution.
        out_dir: str
            Output directory where the results will be saved.
        idx: int
            Index of the algorithm in the Experiment. Userful in case you want to experiment with several algorithms.
        """
        name = "{:02}_{}".format(idx, self.__class__.__name__)
        results.to_parquet(path.join(out_dir, "{}.gz".format(
            name)), compression="gzip", index=True)
        return name


class AnomalyDetectionAlgorithm(Algorithm):
    pass


class AnomalyDetectionThresholdBasedAlgorithm(AnomalyDetectionAlgorithm):

    def __init__(self, threshold_mode: str, threshold_value: float):
        self.threshold_mode = threshold_mode
        self.threshold_value = threshold_value

    def find_anomaly_threshold(self, values: np.ndarray) -> float:
        """
        Finds anomaly threshold for threshold based algorithms.
        3 different approaches are currently implemented.

        Parameters
        ----------
        values: Numpy ndarray
            Results of the algoritm execution.

        Returns
        -------
        float
            Calculated threshold value.
        """
        if self.threshold_mode == 'percentile':
            return np.percentile(values, self.threshold_value)
        elif self.threshold_mode == 'sigma':
            return values.std() * self.threshold_value
        elif self.threshold_mode == 'max':
            return values.max()
        else:
            raise RuntimeError("Invalid threshold_mode configuration.")


class AnomalyDetectionClassBasedAlgorithm(AnomalyDetectionAlgorithm):
    pass

class PretrainedAlgorithm(Algorithm):

    def __init__(self, pretrained: bool, path: str=None) -> None:
        super().__init__()
        if pretrained and path is None:
            # Check path of model weights
            raise RuntimeError("The model seems to have been pretrained but weights path is None")
        self.path = path
        self.pretrained = pretrained

    @abstractmethod
    def load_weights(self) -> None:
        "Load pretrained model/weights for the algorithm's path."
        pass