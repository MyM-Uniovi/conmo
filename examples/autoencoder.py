import os
import sys

# Add package to path (Uncomment only in case you have downloaded Conmo from github repository)
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from conmo.experiment import Experiment, Pipeline
from conmo.algorithms import KerasAutoencoder
from conmo.datasets import NASATurbofanDegradation
from conmo.metrics import Accuracy
from conmo.preprocesses import SklearnPreprocess
from conmo.splitters import SklearnSplitter
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

# Pipeline definition
dataset = NASATurbofanDegradation('FD001')
# TODO: Check splitter operation
splitter = SklearnSplitter(splitter=TimeSeriesSplit(n_splits=2, test_size=0.3))
preprocesses = [
    SklearnPreprocess(to_data=True, to_labels=False,
                      test_set=True, preprocess=MinMaxScaler()),
]
algorithms = [
    KerasAutoencoder()
]
metrics = [
    Accuracy()
]
pipeline = Pipeline(dataset, splitter, preprocesses, algorithms, metrics)


# Experiment definition and launch
experiment = Experiment([pipeline], [])
experiment.launch()