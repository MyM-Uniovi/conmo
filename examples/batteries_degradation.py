import os
import sys

# Add package to path (Uncomment only in case you have downloaded Conmo from github repository)
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from conmo.experiment import Experiment, Pipeline
from conmo.algorithms import PretrainedRandomForest, PretrainedCNN1D, PretrainedMultilayerPerceptron
from conmo.datasets import BatteriesDataset
from conmo.metrics import RMSPE
from conmo.splitters import SklearnSplitter
from sklearn.model_selection import PredefinedSplit

# Pipeline definition
# Change path to our local dataset files, specify chemistry of the batteries (LFP, NCA, NMC) and test set
dataset = BatteriesDataset('/home/lucas/DTW-Li-ion-Diagnosis/', 'LFP', 1)
splitter = SklearnSplitter(splitter=PredefinedSplit(dataset.sklearn_predefined_split()))
preprocesses = None
# Changes the path to the files where the pre-trained models are stored (usually h5, h5py or joblib formats).
algorithms = [
    PretrainedRandomForest(pretrained=True, path='/home/lucas/DTW-Li-ion-Diagnosis/saved/LFP/model-RF.joblib'),
    PretrainedMultilayerPerceptron(pretrained=True, input_len=128, path='/home/lucas/DTW-Li-ion-Diagnosis/saved/LFP/model-MLP.h5'),
    PretrainedCNN1D(pretrained=True, input_len=128, path='/home/lucas/DTW-Li-ion-Diagnosis/saved/LFP/model-CNN.h5')
]
metrics = [
    RMSPE()
]
pipeline = Pipeline(dataset, splitter, preprocesses, algorithms, metrics)


# Experiment definition and launch
experiment = Experiment([pipeline], [])
experiment.launch()
