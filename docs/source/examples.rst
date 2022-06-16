.. _examples:

========
Examples
========

A handful of example experiments can be found in the "examples" directory of the repository. These are listed below:

NASA TurboFan Degradation
=========================
This example can be found in `nasa_cmapss.py` file. The chosen dataset is NASA's Turbofan engine degradation simulation data set. It is a dataset widely used in multivariate time series anomaly detection and condition monitoring problems.
The splitter used is the Sklearn Predefined Split. For more information see the `Scikit-Learn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.PredefinedSplit.html>`_.
Regarding preprocessing, several have been used. The Savitzky-Golay filter, RUL Imputation and Binarizer are already implemented in Conmo. The MinMaxScaler is a Sklearn preprocessing (more information `here <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html>`_) that has been packaged using SklearnPreprocess. Finally, two custom preprocesses for data cleaning and label renaming have been defined using the CustomPreprocess wrapper. To create these preprocesses just create a function that has as parameters the Pandas Dataframes for data and labels.
The algorithms used were dimensionality reduction with PCA together with Mahalanobis distance calculation and One Class Support Vector Machine.
Finally, the metric used was Acurracy.

.. code-block:: python
        :linenos:

        import pandas as pd
        from sklearn.model_selection import PredefinedSplit
        from sklearn.preprocessing import MinMaxScaler

        from conmo import Experiment, Pipeline
        from conmo.algorithms import OneClassSVM, PCAMahalanobis
        from conmo.datasets import NASATurbofanDegradation
        from conmo.metrics import Accuracy
        from conmo.preprocesses import (Binarizer, CustomPreprocess, RULImputation,
                                        SavitzkyGolayFilter, SklearnPreprocess)
        from conmo.splitters import SklearnSplitter

        # First custom preprocess definition
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

        # Second custom preprocess definition
        def rename_labels(data: pd.DataFrame, labels: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
            # Rename labels from 'rul' to 'anomaly'
            labels.rename(columns={'rul': 'anomaly'}, inplace=True)

            return data, labels


        # Select FD001 subdataset of NASA Turbofan Degradation dataset
        dataset = NASATurbofanDegradation(subdataset="FD001")

        # Split dataset using predefined dataset split
        splitter = SklearnSplitter(splitter=PredefinedSplit(dataset.sklearn_predefined_split()))

        # Preprocesses definition
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

        # Algorithms definiition with default parameters
        algorithms = [
            PCAMahalanobis(),
            OneClassSVM()
        ]

        metrics = [
            Accuracy()
        ]
        # Pipeline with all steps
        pipeline = Pipeline(dataset, splitter, preprocesses, algorithms, metrics)

        # Experiment definition and launch
        experiment = Experiment([pipeline], [])
        experiment.launch()

Batteries Degradation
=====================
This experiment can be found in the file `batteries_degradation.py` and reproduces the results obtained in a paper to estimate the level of degradation of some types of lithium batteries.
The dataset used is Batteries Degradation. This is not a time series, although it is somewhat similar since it measures different types of degradation in three types of batteries as they are gradually used. It is a local dataset, so it is necessary to pass the path in which it is located, and also the type of battery to be selected (LFP) and the test set, in this case 1.
The splitter used is the Sklearn Predefined Split and it does not have any preprocessing since during the parsing of the local files to the Conmo format the data is already normalised.
The algorithms used are the same as those used in the paper: Random Forest, Multilayer Perceptron and Convolutional Neural Network. In all cases the pre-trained models are used, so it is necessary to pass the path to the files as a parameter.
The metric used is Root Mean Square Percentage Error.

.. code-block:: python
        :linenos:

        from conmo import Experiment, Pipeline
        from conmo.algorithms import PretrainedRandomForest, PretrainedCNN1D, PretrainedMultilayerPerceptron
        from conmo.datasets import BatteriesDataset
        from conmo.metrics import RMSPE
        from conmo.splitters import SklearnSplitter
        from sklearn.model_selection import PredefinedSplit

        # Pipeline definition
        # Change path to our local dataset files, specify chemistry of the batteries (LFP, NCA, NMC) and test set
        dataset = BatteriesDataset('/path/to/batteries/dataset/', 'LFP', 1)
        splitter = SklearnSplitter(splitter=PredefinedSplit(dataset.sklearn_predefined_split()))
        preprocesses = None
        # Changes the path to the files where the pre-trained models are stored (usually h5, h5py or joblib formats).
        algorithms = [
            PretrainedRandomForest(pretrained=True, path='/path/to/saved/model-RF.joblib'),
            PretrainedMultilayerPerceptron(pretrained=True, input_len=128, path='/path/to/saved/model-MLP.h5'),
            PretrainedCNN1D(pretrained=True, input_len=128, path='/path/to/saved/model-CNN.h5')
        ]
        metrics = [
            RMSPE()
        ]
        pipeline = Pipeline(dataset, splitter, preprocesses, algorithms, metrics)


        # Experiment definition and launch
        experiment = Experiment([pipeline], [])
        experiment.launch()

Server Machine Dataset with PCAMahalanobis
==========================================
This experiment can be found in the file `omni_anomaly_smd.py`.
The Server Machine Dataset used in this experiment has been obtained from the OmniAnomaly repository. In their `Github <https://github.com/NetManAIOps/OmniAnomaly>`_ you can find more information about the dataset as well as the implementation of other anomaly detection and time series data mining algorithms.
The splitter used is the Sklearn Predefined Split and the preprocessing is the MinMaxScaler from Sklearn.
The algorithms is PCA with Mahalanobis distance.
Finally, the metric is the Accuracy.

.. code-block:: python
        :linenos:

        from sklearn.preprocessing import MinMaxScaler

        from conmo import Experiment, Pipeline
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