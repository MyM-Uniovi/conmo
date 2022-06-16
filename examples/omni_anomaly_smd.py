import os
import sys

# Add package to path (Uncomment only in case you have downloaded Conmo from github repository)
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.preprocessing import MinMaxScaler

from conmo.experiment import Experiment, Pipeline
from conmo.algorithms import PCAMahalanobis
from conmo.datasets import ServerMachineDataset
from conmo.metrics import Accuracy
from conmo.preprocesses import SklearnPreprocess
from conmo.splitters import SklearnSplitter
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import MinMaxScaler

# Pipeline definition
dataset = ServerMachineDataset('1-01')
splitter = SklearnSplitter(splitter=PredefinedSplit(dataset.sklearn_predefined_split()))
preprocesses = [
    SklearnPreprocess(to_data=True, to_labels=False,
                      test_set=True, preprocess=MinMaxScaler()),
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
