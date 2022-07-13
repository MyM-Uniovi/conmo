import os
import sys

# Add package to path (Uncomment only in case you have downloaded Conmo from github repository)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from conmo.experiment import Experiment, Pipeline
from conmo.algorithms import SkipGramPerplexity
from conmo.datasets import ServerMachineDataset
from conmo.metrics import Accuracy
from conmo.preprocesses import DiscretizeDataset
from conmo.preprocesses import CustomPreprocess
from conmo.splitters import SklearnSplitter
from sklearn.model_selection import PredefinedSplit

def reduce_columns(data: pd.DataFrame, labels: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    # Delete columns with constant values
    columns = ['udp_snd_buf_errs', 'udp_rcv_buf_errs', 'listen_overflows', 'total_mem', 'mem_shmem', 'si', 'so', 'in_errs']
    sub_data = data.drop(columns=columns, axis=1)

    return sub_data, labels

# Pipeline definition
dataset = ServerMachineDataset('1-01')
splitter = SklearnSplitter(splitter=PredefinedSplit(dataset.sklearn_predefined_split()))
preprocesses = [
    CustomPreprocess(reduce_columns),
    DiscretizeDataset(to_data=True, to_labels=False, test_set=True)
]
algorithms = [
    SkipGramPerplexity(epochs=1, embed_size=100)
]
metrics = [
    Accuracy()
]
pipeline = Pipeline(dataset, splitter, preprocesses, algorithms, metrics)


# Experiment definition and launch
experiment = Experiment([pipeline], [])
experiment.launch()