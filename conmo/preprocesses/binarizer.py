from typing import Iterable, Union

import pandas as pd

from conmo.conf import Index
from conmo.preprocesses.preprocess import ExtendedPreprocess


class Binarizer(ExtendedPreprocess):

    def __init__(self, to_data: Union[bool, Iterable[str]], to_labels: Union[bool, Iterable[str]], test_set: bool, threshold: int) -> None:
        super().__init__(to_data, to_labels, test_set)
        self.threshold = threshold

    def transform(self, df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        # Binarize columns, setting TRUE if value is less than threshold and FALSE if values is equal or greather than threshold
        if self.test_set == True:
            index_slice = pd.IndexSlice[:, columns]
        else:
            # Only TRAIN sequences
            index_slice = pd.IndexSlice[(
                slice(None), Index.SET_TRAIN), columns]

        df.loc[index_slice] = df.loc[index_slice] < self.threshold

        return df
