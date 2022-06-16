from abc import ABC, abstractmethod
from os import path
from typing import Iterable

import pandas as pd

from conmo.conf import File, Index, Label


class Metric(ABC):

    @abstractmethod
    def calculate(self, idx: int, algorithms: Iterable[str], last_preprocess_dir: str, algorithms_dir: str, metrics_dir: str) -> None:
        """
        Calculates specific metric for each of the algorithms' results.

        Parameters
        ----------
        idx: str
            Index of the metric in the Experiment. Userful in case you want to calculate several metrics.
        algoritmss: Iterable[str]
            List of names of the selected algorithms.
        last_preprocess_dir:
            Name of the directory where the ground truth is located
        algorithms_dir:
            Name of the directory where the results of the algorithms executions are stored.
        metrics_dir:
            Name of th edirectory where the results will be stored.
        """
        pass

    def problem_label(self, truth: pd.DataFrame) -> str:
        """
        Determinates the nature of the problem by identifying the column's name of the labels.

        Parameters
        ----------
        truth: Pandas Dataframe
            Labels file of the dataset.
        Returns
        -------
        str
            Returns the column for the metric.

        Raises
        ------
        RuntimeError
            If the labels of the ground truth are invalid for the problem.
        """
        if Label.ANOMALY in truth.columns:
            return Label.ANOMALY
        elif Label.RUL in truth.columns:
            return Label.RUL
        elif Label.BATTERIES_DEG_TYPES == truth.columns.values.tolist(): # It's a problem related with batteries degradation
            return Label.BATTERIES_DEG_TYPES
        else:
            raise RuntimeError("Invalid labels for ground truth.")

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
        # First level is FOLD
        if labels.index.nlevels == 2 and labels.index.names[1] == Index.SEQUENCE:
            return True
        elif labels.index.nlevels == 3 and labels.index.names[1] == Index.SEQUENCE and labels.index.names[2] == Index.TIME:
            return False
        else:
            raise RuntimeError("Invalid number of levels for labels.")

    def show_start_message(self):
        """
        Simple method to print on the terminal the name of the used metric.
        """
        print("\n+++ Metric {} +++".format(self.__class__.__name__))

    def load_truth(self, last_preprocess_dir: str):
        """
        Load labels from the last preprocess directory.

        Parameters
        ----------
        last_preprocess_dir: str
            Last diretory where the labels dataframe was stored.

        Returns
        -------
        Pandas Dataframe
            Dataframe cantainig the labels.
        """
        return pd.read_parquet(path.join(last_preprocess_dir, File.LABELS))

    def load_results(self, algorithm: str, algorithms_dir: str) -> pd.DataFrame:
        """
        Load results for a specific algorthm.

        Parameters
        ----------
        algoritm: str
            Name of the selected algorithm.
        algorithms_dir: str
            Name of the directory where the results of the algorithms executions are stored.

        Returns
        -------
        Pandas Dataframe
            Dataframe cantainig the results (predictions).
        """
        return pd.read_parquet(path.join(algorithms_dir, "{}.gz".format(algorithm)))

    def save_output(self, metric: pd.DataFrame, idx: int, metrics_dir: str) -> None:
        """
        Save metric's output to disk.

        Parameters
        ----------
        metric: Pandas Dataframe
            Dataframe containing the metric's results. 
        idx: int
            Index of the metric in the Experiment. Userful in case you want to calculate several metrics.
        metrics_dir: str
            Name of the directory where the results will be stored.
        """
        name = "{:02}_{}".format(idx, self.__class__.__name__)
        print(metric)
        metric.to_parquet(path.join(metrics_dir, "{}.gz".format(
            name)), compression="gzip", index=True)
