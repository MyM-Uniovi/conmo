import os
import sys

# Add package to path (Uncomment only in case you have downloaded Conmo from github repository)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import StandardScaler

from conmo.experiment import Experiment, Pipeline
from conmo.algorithms import PCAMahalanobis
from conmo.datasets import NASATurbofanDegradation
from conmo.metrics import Accuracy
from conmo.preprocesses import (CustomPreprocess, RULImputation,
                                SequenceWindowing, SimpleExponentialSmoothing)
from conmo.splitters import SklearnSplitter

config = {
    'dataset': 'FD001',
    'sensors': ['T30', 'T50', 'P30', 'Ps30', 'phi'],
    'sequence_length': 30,
    'alpha': 0.1,
    'threshold': 125
}


def condition_scaler(data: pd.DataFrame, labels: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    # Generate operating condition as categorical variable
    data.loc[:, 'op_cond'] = abs(data.loc[:, 'setting_1'].round()).astype(str) + "_" + abs(
        data.loc[:, 'setting_2'].round(decimals=2)).astype(str) + "_" + data.loc[:, 'TRA'].astype(str)

    # Remove unnecessary variables
    columns = ['op_cond']
    columns.extend(config['sensors'])
    data = data.loc[:, columns]

    # Apply operating condition specific scaling
    scaler = StandardScaler()
    for condition in data.loc[:, 'op_cond'].unique():
        # Only fit with TRAIN data
        scaler.fit(data.loc[(data.index.get_level_values('set') == 'train') & (
            data['op_cond'] == condition), config['sensors']])

        # Transform both TRAIN and TEST data
        data.loc[data['op_cond'] == condition, config['sensors']] = scaler.transform(
            data.loc[data['op_cond'] == condition, config['sensors']])

    # Remove 'op_cond' variable
    data.drop(columns=['op_cond'], inplace=True)

    return data, labels


# Pipeline definition
dataset = NASATurbofanDegradation(subdataset=config['dataset'])
splitter = SklearnSplitter(PredefinedSplit(dataset.sklearn_predefined_split()))
preprocesses = [
    # Scale with respect to the operating condition
    CustomPreprocess(condition_scaler),
    # Generate RUL values according to the piece-wise target function
    RULImputation(threshold=config['threshold']),
    # Exponential smoothing
    # SimpleExponentialSmoothing(
    #     to_data=True, to_labels=False, test_set=True, alpha=config['alpha'], adjust=True),
    # Windowing
    SequenceWindowing(window_length=30, augment_train=True, augment_test=False, fill_value=-99.9)
]
algorithms = [
    PCAMahalanobis()
]
metrics = [
    Accuracy()
]
pipeline = Pipeline(dataset, splitter, preprocesses, algorithms, metrics)


# Experiment definition and launch
experiment = Experiment([pipeline], [])
experiment.launch()
