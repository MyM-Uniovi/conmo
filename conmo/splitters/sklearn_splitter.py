from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold,
                                     LeaveOneGroupOut, LeaveOneOut,
                                     LeavePGroupsOut, LeavePOut,
                                     PredefinedSplit, RepeatedKFold,
                                     RepeatedStratifiedKFold, ShuffleSplit,
                                     StratifiedKFold, StratifiedShuffleSplit,
                                     TimeSeriesSplit)

from conmo.conf import Index
from conmo.splitters.splitter import Splitter


class SklearnSplitter(Splitter):

    def __init__(self, splitter: Union[GroupKFold, GroupShuffleSplit, KFold, LeaveOneGroupOut, LeavePGroupsOut, LeaveOneOut, LeavePOut, PredefinedSplit, RepeatedKFold, RepeatedStratifiedKFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit, TimeSeriesSplit], groups: Optional[Iterable[int]] = None) -> None:
        self.splitter = splitter
        self.groups = groups

    def split(self, in_dir: str, out_dir: str) -> None:
        self.show_start_message()
        data, labels = self.load_input(in_dir)

        # Check previous splitting
        if self.already_splitted(data) == True or self.already_splitted(labels) == True:
            raise RuntimeError("Dataset already splitted.")

        # Extract sequences of data (both DATA and LABELS must be equal)
        if (data.index.get_level_values(Index.SEQUENCE).unique() != labels.index.get_level_values(Index.SEQUENCE).unique()).any():
            raise RuntimeError(
                "Sequence indexes of DATA and LABELS does not match.")
        sequences = data.index.get_level_values(Index.SEQUENCE).unique()

        # Split data by calling sklearn split function over sequences
        data_col = []
        data_index = []
        labels_col = []
        labels_index = []
        for fold_idx, (train_idx, test_idx) in enumerate(self.splitter.split(sequences, groups=self.groups)):
            # Sklearn split function returns indexes of sequences, not sequences directly, so must access their value first (sequences[idx])
            # This loop provides both FOLD (fold_idx) of the current split and the TRAIN/TEST (train_idx, test_idx) sets

            # DATA
            data_col_fold, data_index_fold = self.extract_fold(
                data, sequences, fold_idx+1, train_idx, test_idx)
            data_col.extend(data_col_fold)
            data_index.extend(data_index_fold)

            # LABELS
            labels_col_fold, labels_index_fold = self.extract_fold(
                labels, sequences, fold_idx+1, train_idx, test_idx)
            labels_col.extend(labels_col_fold)
            labels_index.extend(labels_index_fold)

        # Generate output DataFrames
        data = self.to_dataframe(data, data_col, data_index)
        labels = self.to_dataframe(labels, labels_col, labels_index)
        self.save_output(out_dir, data, labels)

    def extract_fold(self, df: pd.DataFrame, sequences: np.ndarray, fold: int, train_idx: np.ndarray, test_idx: np.ndarray) -> (np.ndarray, np.ndarray):
        # Generate array of column values of the fold
        data = np.concatenate([df.loc[sequences[train_idx], :].to_numpy(
        ), df.loc[sequences[test_idx], :].to_numpy()])

        # Extract number of samples of each sequence (1 if no 'time' index is present) (1D array of each sequence count)
        train_samples = df.loc[sequences[train_idx],
                               :].groupby(level=Index.SEQUENCE).size().to_numpy()
        test_samples = df.loc[sequences[test_idx],
                              :].groupby(level=Index.SEQUENCE).size().to_numpy()

        df_index = df.index.to_frame()
        # Generate levels of the new dataframe individually
        # Time level, if present, must contain the original values
        if Index.TIME in df_index:
            time_level = np.concatenate([df_index.loc[sequences[train_idx], Index.TIME].to_numpy(
            ), df_index.loc[sequences[test_idx], Index.TIME].to_numpy()])
        # Sequence, Set and Fold levels must be generated for each fold
        sequence_level = np.repeat([np.arange(
            1, train_idx.shape[0]+1), np.arange(1, test_idx.shape[0]+1)], np.concatenate([train_samples, test_samples]))
        # Convert 1D array to unique value
        train_samples = train_samples.sum()
        test_samples = test_samples.sum()
        # Set and Fold levels
        set_level = np.repeat([Index.SET_TRAIN, Index.SET_TEST], [
                              train_samples, test_samples])
        fold_level = np.repeat(fold, train_samples + test_samples)

        # Generate array of index values of the fold
        if Index.TIME in df_index:
            index = np.concatenate((fold_level.reshape((-1,1)), set_level.reshape((-1,1)), sequence_level.reshape((-1,1)), time_level.reshape((-1,1))), axis=1, dtype=object)
        else:
            index = np.concatenate((fold_level.reshape((-1,1)), set_level.reshape((-1,1)), sequence_level.reshape((-1,1))), axis=1, dtype=object)

        return data, index

    def to_dataframe(self, df: pd.DataFrame, data: np.ndarray, index: np.ndarray) -> pd.DataFrame:
        # Generate MultiIndex
        columns = [Index.FOLD, Index.SET]
        columns.extend(df.index.names)  # To include previous index columns
        index = pd.DataFrame(index, columns=columns)

        # Generate DataFrame
        df = pd.DataFrame(data, columns=df.columns,
                          index=pd.MultiIndex.from_frame(index))
        df.sort_index(inplace=True)
        return df
