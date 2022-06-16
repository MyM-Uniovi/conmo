from typing import Iterable, Union

import pandas as pd

from conmo.conf import Index
from conmo.preprocesses.preprocess import ExtendedPreprocess


class SimpleExponentialSmoothing(ExtendedPreprocess):

    def __init__(self, to_data: Union[bool, Iterable[str]], to_labels: Union[bool, Iterable[str]], test_set: bool, alpha: float, adjust: bool = False):
        super().__init__(to_data, to_labels, test_set)
        self.alpha = alpha
        self.adjust = adjust

    def transform(self, df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        # TODO Esto hay que cambiarlo. No se está teniendo en cuenta la division en train/test, folds, etc... Asi como el valor de test_set
        # return df.groupby(Index.SEQUENCE).apply(lambda x: x.ewm(alpha=self.alpha, adjust=self.adjust).mean())
        raise NotImplementedError("TODO")
