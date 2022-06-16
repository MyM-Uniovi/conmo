from abc import ABC, abstractmethod
from os import path
from typing import Iterable, Union

import pandas as pd

from conmo.conf import File, Index


class Preprocess(ABC):
    """
    Abstract base class for a Preprocess.

    This class is an abstract class from which other subclasses inherit and must not be instanciated directly.
    """

    @abstractmethod
    def apply(self, in_dir: str, out_dir: str) -> None:
        """
        Applies the preprocess to the given dataset.

        Parameters
        ----------
        in_dir: str
            Input directory where the files are located. Usually, this is the output directory of the splitter step.
        out_dir: str
            Output directory where the files will be saved.
        """
        pass

    def show_start_message(self) -> None:
        """
        Simple method to print on the terminal the name of the selected splitter.
        """
        print("\n+++ Preprocess {} +++".format(self.__class__.__name__))

    def load_input(self, in_dir: str) -> (pd.DataFrame, pd.DataFrame):
        """
        Read parquet data and labels files of the chosen dataset before it's split.
        
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

    def save_output(self, out_dir: str, data: pd.DataFrame, labels: pd.DataFrame) -> None:
        """
        Save preprocessed dataset to parquet format.

        Parameters
        ----------
        out_dir: str
            Output directory where the results will be saved.
        data: Pandas Dataframe
            Preprocessed data.
        labels: Pandas Dataframe
            Preprocessed labels.
        """
        data.to_parquet(path.join(out_dir, File.DATA),
                        compression="gzip", index=True)
        labels.to_parquet(path.join(out_dir, File.LABELS),
                          compression="gzip", index=True)


class ExtendedPreprocess(Preprocess):
    """
    Specific class to implement preprocessing which consists of applying certain transformations on some columns of the dataset. 
    The preprocessing that inherit from this class have in the constructor to_data, to_labels and test_set to indicate the columns
    on which to apply the DATA and LABELS preprocessing respectively, and if the TEST ones are included or not.
    """

    def __init__(self, to_data: Union[bool, Iterable[str]], to_labels: Union[bool, Iterable[str]], test_set: bool) -> None:
        self.to_data = to_data
        self.to_labels = to_labels
        self.test_set = test_set

    @abstractmethod
    def transform(self, df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        """
        Performs the preprocess over the dataframe with the given columns.

        Parameters
        ----------
        df: Pandas Dataframe
            Dataframe containing the data or the labels of the dataset.
        columns: Iterable[str]
            List of columns that will be used in the preprocess. Also the columns of the final dataframe.

        Returns
        -------
        Pandas Dataframe:
            Dataframe preprocessed.
        """
        pass

    def apply(self, in_dir: str, out_dir: str) -> None:
        self.show_start_message()
        data, labels = self.load_input(in_dir)

        # DATA
        if self.to_data != False:
            data = self.transform(
                data, self.extract_columns(data, self.to_data))

        # LABELS
        if self.to_labels != False:
            labels = self.transform(
                labels, self.extract_columns(labels, self.to_labels))

        self.save_output(out_dir, data, labels)

    def extract_columns(self, df: pd.DataFrame, columns: Union[bool, Iterable[str]]) -> Iterable[str]:
        """
        Returns a list containig all the column's name of the data.

        Parameters
        ----------
            df: Pandas Dataframe
                Dataframe containing the data.
            columns: Union[bool, Iterable[str]]
                Bool value if the dataframe has columns or the list of columns.

        Returns
        -------
        columns: Iterable[str]
            List containing the names of the dataframe's columns.
        """
        if columns == True:
            return df.columns
        else:
            return columns
