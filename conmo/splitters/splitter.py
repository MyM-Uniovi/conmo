from abc import ABC, abstractmethod
from os import path

import pandas as pd

from conmo.conf import File, Index


class Splitter(ABC):

    @abstractmethod
    def split(self, in_dir: str, out_dir: str) -> None:
        """
        Performs the split to both data and labels of the dataset.

        Parameters
        ----------
        in_dir: str
            Input directory of the before step.
        out_dir: str
            Output directory where te split data will be stored.
        """
        pass

    def show_start_message(self):
        """
        Simple method to print on the terminal the name of the selected splitter.
        """
        print("\n+++ Splitter {} +++".format(self.__class__.__name__))

    def already_splitted(self, df: pd.DataFrame) -> bool:
        """
        Checks if the dataset was already splitted.

        Parameters
        ----------
        df: Pandas Dataframe
            Input dataset.

        Returns
        -------
        bool
            True in case the dataset was already splitted, False otherwise.

        Raises
        ------
        RuntimeError
            If the dataset isn't splitted and doesn't follow Conmo's format.

        """
        nindex = df.index.names
        # Soft comparison to allow both [SEQUENCE,TIME] and only [TIME] indexes for DATA and LABELS dataframes
        if (len(nindex) == 3 or len(nindex) == 4) and nindex[0] == Index.FOLD and nindex[1] == Index.SET and nindex[2] == Index.SEQUENCE:
            return True
        elif (len(nindex) == 1 or len(nindex) == 2) and nindex[0] == Index.SEQUENCE:
            return False
        else:
            print(nindex)
            raise RuntimeError(
                "Input DataFrame does not contain a valid index configuration.")

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
        
        Raises
        ------
        If data and labels have different sequences values.
        """
        # Load input dataframes
        data = pd.read_parquet(path.join(in_dir, File.DATA))
        labels = pd.read_parquet(path.join(in_dir, File.LABELS))

        # Check both DATA and LABELS have the same sequences indexes
        if not data.index.get_level_values(Index.SEQUENCE).unique().equals(labels.index.get_level_values(Index.SEQUENCE).unique()):
            raise RuntimeError(
                "Data and Labels files have different sequences values. Both must have the same values")

        return data, labels

    def save_output(self, out_dir: str, data: pd.DataFrame, labels: pd.DataFrame) -> None:
        """
        Save splitted dataset to parquet format.

        Parameters
        ----------
        out_dir: str
            Output directory where the results will be saved.
        data: Pandas Dataframe
            Splitted data.
        labels: Pandas Dataframe
            Splitted labels.
        """
        # Save output dataframes
        data.to_parquet(path.join(out_dir, File.DATA),
                        compression="gzip", index=True)
        labels.to_parquet(path.join(out_dir, File.LABELS),
                          compression="gzip", index=True)
