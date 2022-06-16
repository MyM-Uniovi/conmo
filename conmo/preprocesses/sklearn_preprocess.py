from typing import Iterable, Union

import pandas as pd
from sklearn.preprocessing import (Binarizer, FunctionTransformer,
                                   KBinsDiscretizer, KernelCenterer,
                                   LabelBinarizer, LabelEncoder, MaxAbsScaler,
                                   MinMaxScaler, MultiLabelBinarizer,
                                   Normalizer, OneHotEncoder, OrdinalEncoder,
                                   PolynomialFeatures, PowerTransformer,
                                   QuantileTransformer, RobustScaler,
                                   StandardScaler)

from conmo.conf import Index
from conmo.preprocesses.preprocess import ExtendedPreprocess


class SklearnPreprocess(ExtendedPreprocess):
    """
    Class used to wrap existing preprocess in the Scikit-Learn library.
    It also allows this preprocess to be applied to certain columns of the dataset.
    """

    def __init__(self, to_data: Union[bool, Iterable[str]], to_labels: Union[bool, Iterable[str]], test_set: bool, preprocess: Union[Binarizer, FunctionTransformer, KBinsDiscretizer, KernelCenterer, LabelBinarizer, LabelEncoder, MultiLabelBinarizer, MaxAbsScaler, MinMaxScaler, Normalizer, OneHotEncoder, OrdinalEncoder, PolynomialFeatures, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler]) -> None:
        super().__init__(to_data, to_labels, test_set)
        self.preprocess = preprocess

    def transform(self, df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        # Fit with TRAIN data and transform both TRAIN and TEST, fold by fold
        for fold in df.index.get_level_values(Index.FOLD).unique():
            # Train: fit and transform
            index_slice = pd.IndexSlice[(fold, Index.SET_TRAIN), columns]
            df.loc[index_slice] = self.preprocess.fit_transform(
                df.loc[index_slice].values)

            # Test: transform
            if self.test_set == True:
                index_slice = pd.IndexSlice[(fold, Index.SET_TEST), columns]
                df.loc[index_slice] = self.preprocess.transform(
                    df.loc[index_slice].values)

        return df
