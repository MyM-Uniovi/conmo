.. _api:

=============
API Reference
=============

This is the API Reference documentation of the package, including modules, classes and functions.

:mod:`conmo.experiments`
========================

.. automodule:: conmo.experiment

This is the main submodule of the package and it is the responsible of create the intermediary directories of the experiment and take care of creating and executing the configured pipeline.

.. currentmodule:: conmo

.. autosummary::
    :template: classtemplate.rst
    :toctree: modules/experiments

    experiment.Experiment
    experiment.Pipeline

:mod:`conmo.datasets`
=====================

.. automodule:: conmo.datasets

The :mod:`conmo.datasets` submodule takes care of downloading the dataset and parsing it to the Conmo's format.

.. currentmodule:: conmo

.. autosummary::
    :template: classtemplate.rst
    :toctree: modules/datasets
    
    datasets.dataset.Dataset
    datasets.dataset.RemoteDataset
    datasets.dataset.LocalDataset
    datasets.MarsScienceLaboratoryMission
    datasets.SoilMoistureActivePassiveSatellite
    datasets.ServerMachineDataset
    datasets.NASATurbofanDegradation
    datasets.BatteriesDataset

:mod:`conmo.splitters`
=====================

.. automodule:: conmo.splitters

Once the dataset has been loaded, it is necessary to separate the training and test parts. The :mod:`conmo.splitters` submodule permits generate new splitters or use predefined ones from the Scikit-Learn library.

.. currentmodule:: conmo

.. autosummary::
    :template: classtemplate.rst
    :toctree: modules/splitters
    
    splitters.splitter.Splitter
    splitters.SklearnSplitter


:mod:`conmo.preprocesses`
=========================

.. automodule:: conmo.preprocesses

The aim of the :mod:`conmo.preprocesses` submodule is to apply a series of transformations to the data set before it is used as input to the algorithms. Several types of preprocesses implemented are usually used in time series anomaly detection problems.

.. currentmodule:: conmo

.. autosummary::
    :template: classtemplate.rst
    :toctree: modules/preprocesses

    preprocesses.preprocess.Preprocess
    preprocesses.preprocess.ExtendedPreprocess
    preprocesses.Binarizer
    preprocesses.CustomPreprocess
    preprocesses.RULImputation
    preprocesses.SavitzkyGolayFilter
    preprocesses.SklearnPreprocess

:mod:`conmo.algorithms`
=======================

.. automodule:: conmo.algorithms

The :mod:`conmo.algorithms` submodule contains everything related to algorithms in Conmo, from abstract classes to introduce new algorithms in Conmo to implementations of some of the algorithms used in the example experiments.

.. currentmodule:: conmo

.. autosummary::
    :template: classtemplate.rst
    :toctree: modules/algorithms/anomaly_detection

    algorithms.algorithm.Algorithm
    algorithms.algorithm.AnomalyDetectionThresholdBasedAlgorithm
    algorithms.algorithm.AnomalyDetectionClassBasedAlgorithm
    algorithms.PCAMahalanobis
    algorithms.OneClassSVM
    algorithms.KerasAutoencoder
    algorithms.PretrainedRandomForest
    algorithms.PretrainedMultilayerPerceptron
    algorithms.PretrainedCNN1D

:mod:`conmo.metrics`
====================

.. automodule:: conmo.metrics

The :mod:`conmo.metrics` submodule contains everything necessary to add new ways of measuring the effectiveness of the implemented algorithms. Accuracy and RMSPE are currently implemented.

.. currentmodule:: conmo

.. autosummary::
    :template: classtemplate.rst
    :toctree: modules/metrics
    
    metrics.metric.Metric
    metrics.Accuracy
    metrics.RMSPE

