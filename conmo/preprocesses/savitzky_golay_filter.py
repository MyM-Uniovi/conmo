from typing import Iterable, Optional, Union

import pandas as pd
from scipy.signal import savgol_filter

from conmo.conf import Index
from conmo.preprocesses.preprocess import ExtendedPreprocess


class SavitzkyGolayFilter(ExtendedPreprocess):

    def __init__(self, to_data: Union[bool, Iterable[str]], to_labels: Union[bool, Iterable[str]], test_set: bool, window_length: int, polyorder: int, deriv: Optional[int] = 0, delta: Optional[float] = 1.0, mode: Optional[str] = 'interp', cval: Optional[float] = 0.0):
        super().__init__(to_data, to_labels, test_set)
        if window_length % 2 == 0:
            raise RuntimeError(
                'The window_length must be a positive odd integer.')
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta
        self.mode = mode
        self.cval = cval

    def transform(self, df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        if self.test_set == True:
            # Select all sequences
            groups = df.groupby(
                level=[Index.FOLD, Index.SET, Index.SEQUENCE])
        else:
            # Select only TRAIN sequences
            groups = df.loc[(slice(None), Index.SET_TRAIN), :].groupby(
                level=[Index.FOLD, Index.SET, Index.SEQUENCE])

        for group in groups:
            for column in columns:
                # Apply SavGolFilter for each column of each sequence
                df.loc[group[0], column] = savgol_filter(
                    x=group[1].loc[:,column].to_numpy(), window_length=self.window_length, polyorder=self.polyorder, deriv=self.deriv, delta=self.delta, mode=self.mode, cval=self.cval)

        return df
