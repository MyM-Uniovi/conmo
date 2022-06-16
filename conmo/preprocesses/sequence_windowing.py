from typing import Any

import numpy as np
import pandas as pd

from conmo.conf import Index
from conmo.preprocesses.preprocess import Preprocess


class SequenceWindowing(Preprocess):

    def __init__(self, window_length: int, augment_train: bool, augment_test: bool, fill_value: Any):
        self.window_length = window_length
        self.augment_train = augment_train
        self.augment_test = augment_test
        self.fill_value = fill_value

    def apply(self, in_dir, out_dir) -> None:
        self.show_start_message()
        data, labels = self.load_input(in_dir)

        # Perform sequence windowing
        data = self.dataframe_windowing(data)
        labels = self.dataframe_windowing(labels)

        self.save_output(out_dir, data, labels)

    def dataframe_windowing(self, df: pd.DataFrame) -> pd.DataFrame:
        # Data and Index must be generated separately due to performance
        train_data, train_index = self.data_index_generator(
            df.xs(Index.SET_TRAIN, level=Index.SET, drop_level=False), self.augment_train)
        test_data, test_index = self.data_index_generator(
            df.xs(Index.SET_TEST, level=Index.SET, drop_level=False), self.augment_test)

        # Update Index to ensure all values are unique
        index = self.update_sequences(pd.DataFrame(np.concatenate(
            [train_index, test_index]), columns=df.index.to_frame().columns))

        # Generate output DataFrame and sort Index
        df = pd.DataFrame(np.concatenate(
            [train_data, test_data]), columns=df.columns, index=pd.MultiIndex.from_frame(index))
        df.sort_index(inplace=True)

        return df

        # Generate DATA sequences
        # data = self.sliding_window(data)
        # labels = self.sliding_window(labels)

        # data_train = np.concatenate(list(sequence for group in data.loc[(slice(None), Index.SET_TRAIN), columns].groupby(
        #     level=[Index.FOLD, Index.SET, Index.SEQUENCE]) for sequence in list(sequence_window(group[1].to_numpy(), self.augment_train, True))))
        # data_test = np.concatenate(list(sequence for group in data.loc[(slice(None), Index.SET_TEST), columns].groupby(
        #     level=[Index.FOLD, Index.SET, Index.SEQUENCE]) for sequence in list(sequence_window(group[1].to_numpy(), self.window_length, self.augment_test, self.fill_value, True))))

        # # Generate indexes as numpy array for optimum performance
        # index_train = np.concatenate(list(sequence for group in group_by_sequence(df.index.to_frame().loc[(slice(
        #     None), 'train'), :]) for sequence in list(sequence_window(group[1].to_numpy(), self.window_length, True, fill_value, False))))
        # index_test = np.concatenate(list(sequence for group in group_by_sequence(df.index.to_frame().loc[(slice(
        #     None), 'test'), :]) for sequence in list(sequence_window(group[1].to_numpy(), self.window_length, augment_test, fill_value, False))))

        # # Update index sequences values to ensure all of them are unique and sorted
        # index = update_sequences(pd.DataFrame(np.concatenate(
        #     [index_train, index_test]), columns=df.index.to_frame().columns), self.window_length)

        # return pd.DataFrame(np.concatenate([data_train, data_test]), columns=columns, index=pd.MultiIndex.from_frame(index))

    def data_index_generator(self, df: pd.DataFrame, augment: bool) -> (np.ndarray, np.ndarray):
        # Generate sequences for data grouping by 'sequence'
        data = np.concatenate(list(sequence for group in df.groupby(
            level=[Index.FOLD, Index.SET, Index.SEQUENCE]) for sequence in list(self.sequence_generator(group[1].to_numpy(), augment, True))))

        # Generate sequences for indexes grouping by 'sequence'
        index = np.concatenate(list(sequence for group in df.index.to_frame().groupby(
            level=[Index.FOLD, Index.SET, Index.SEQUENCE]) for sequence in list(self.sequence_generator(group[1].to_numpy(), augment, False))))

        return data, index

    def sequence_generator(self, array: np.ndarray, augment: bool, is_data: bool):
        """
        Generator for sequences of 'window' size, using sliding windows.
        """
        if array.shape[0] < self.window_length:
            if is_data == True:
                # Data source is smaller than window length. Data imputation is needed
                padding = np.full_like(array, fill_value=self.fill_value, shape=(
                    self.window_length - array.shape[0], array.shape[1]))
                array = np.concatenate([padding, array])
            else:
                # Index source is smaller than window length. New indexes must be generated and time values must be updated
                array = np.repeat([array[0]], self.window_length, axis=0)
                array[:, array.shape[1]] = np.arange(
                    start=1, stop=self.window_length+1)

        # Generate windows of size 'window'
        if augment == True:
            # Generate as much sequences as possible
            for start in range(array.shape[0] - self.window_length + 1):
                yield array[start:start+self.window_length]
        else:
            # Generate only the last sequence
            start = array.shape[0] - self.window_length
            yield array[start:start+self.window_length]

    def update_sequences(self, df: pd.DataFrame) -> int:
        # Generate index based on columns values (inverse of operation df.index.to_frame())
        df.index = pd.MultiIndex.from_frame(df)

        # Update sequences values for each combination of fold/set, starting from 1, for sequences of fixed length
        for group in df.groupby(level=[Index.FOLD, Index.SET]):
            df.loc[group[0], Index.SEQUENCE] = np.repeat(np.arange(
                start=1, stop=group[1].shape[0] / self.window_length + 1, dtype=int), self.window_length)

        return df


# def sliding_window_unified(df: pd.DataFrame) -> pd.DataFrame:
#     # Generate COLUMNS sequences (values of the columns)
#     col = np.concatenate(
#         list(sequence for group in df.groupby(level=Index.SEQUENCE)))


# def sliding_window_splitted(df: pd.DataFrame) -> pd.DataFrame:
#     # Generate COLUMNS sequences (values of the columns)
#     col_train = np.concatenate(list(sequence for group in df.xs(Index.SET_TRAIN, level=Index.SET, drop_level=False).groupby(
#         level=[Index.FOLD, Index.SET, Index.SEQUENCE]) for sequence in list(sequence_window(group[1].to_numpy(), self.window_length, self.augment_train, self.fill_value, True))))
#     col_test = np.concatenate(list(sequence for group in df.xs(Index.SET_TEST, level=Index.SET, drop_level=False).groupby(
#         level=[Index.FOLD, Index.SET, Index.SEQUENCE]) for sequence in list(sequence_window(group[1].to_numpy(), self.window_length, self.augment_test, self.fill_value, True))))

#     # Generate INDEX sequences (values of the index)
#     index_train = np.concatenate(list(sequence for group in df.index.to_frame().xs(Index.SET_TRAIN, level=Index.SET, drop_level=False).groupby(
#         level=[Index.FOLD, Index.SET, Index.SEQUENCE]) for sequence in list(sequence_window(group[1].to_numpy(), self.window_length, self.augment_train, self.fill_value, False))))
#     index_test = np.concatenate(list(sequence for group in df.index.to_frame().xs(Index.SET_TEST, level=Index.SET, drop_level=False).groupby(
#         level=[Index.FOLD, Index.SET, Index.SEQUENCE]) for sequence in list(sequence_window(group[1].to_numpy(), self.window_length, self.augment_test, self.fill_value, False))))

#     # Update INDEX sequences values to ensure all of them are unique and sorted
#     index = update_sequences(pd.DataFrame(np.concatenate(
#         [index_train, index_test]), columns=df.index.to_frame().columns), self.window_length)

#     # Generate DataFrame with new sequences and indexes
#     return pd.DataFrame(np.concatenate([data_train, data_test]), columns=columns, index=pd.MultiIndex.from_frame(index))
