import os
import sys

# Add package to path (Uncomment only in case you have downloaded Conmo from github repository)
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import MinMaxScaler

from conmo.experiment import Experiment, Pipeline
from conmo.algorithms import OneClassSVM, PCAMahalanobis
from conmo.datasets import NASATurbofanDegradation
from conmo.metrics import Accuracy
from conmo.preprocesses import (Binarizer, CustomPreprocess, RULImputation,
                                SavitzkyGolayFilter, SklearnPreprocess)
from conmo.splitters import SklearnSplitter


def data_cleanup(data: pd.DataFrame, labels: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    # Reduce columns
    columns = ['T30', 'T50', 'P30']
    sub_data = data.loc[:, columns]

    # Rename columns
    sub_data = sub_data.rename(columns={'T50': 'TGT'})

    # Calculate FF
    sub_data.loc[:, 'FF'] = data.loc[:, 'Ps30'] * data.loc[:, 'phi']
    sub_data.head()

    return sub_data, labels


def rename_labels(data: pd.DataFrame, labels: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    # Rename labels from 'rul' to 'anomaly'
    labels.rename(columns={'rul': 'anomaly'}, inplace=True)

    return data, labels


# Select FD001 subdataset of NASA Turbofan Degradation dataset
dataset = NASATurbofanDegradation(subdataset="FD001")

# Split dataset using predefined dataset split
splitter = SklearnSplitter(splitter=PredefinedSplit(dataset.sklearn_predefined_split()))

preprocesses = [
    CustomPreprocess(data_cleanup),
    SklearnPreprocess(to_data=True, to_labels=False,
                      test_set=True, preprocess=MinMaxScaler()),
    SavitzkyGolayFilter(to_data=True, to_labels=False,
                        test_set=True, window_length=7, polyorder=2),
    RULImputation(threshold=125),
    Binarizer(to_data=False, to_labels=[
                     'rul'], test_set=True, threshold=50),
    CustomPreprocess(rename_labels)
]
algorithms = [
    PCAMahalanobis(),
    OneClassSVM()
]
metrics = [
    Accuracy()
]
pipeline = Pipeline(dataset, splitter, preprocesses, algorithms, metrics)


# Experiment definition and launch
experiment = Experiment([pipeline], [])
experiment.launch()
