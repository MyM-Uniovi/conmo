from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.decomposition import PCA

from conmo.conf import Index, Label
from conmo.algorithms.algorithm import AnomalyDetectionThresholdBasedAlgorithm


class PCAMahalanobis(AnomalyDetectionThresholdBasedAlgorithm):

    def __init__(self, n_components: float = 0.95, robust_estimator: bool = False, threshold_mode: str = 'chi2', threshold_value: Union[int, float, None] = 0.95):
        super().__init__(threshold_mode, threshold_value)
        self.n_components = n_components
        self.robust_estimator = robust_estimator

    def fit_predict(self, data_train: pd.DataFrame, data_test: pd.DataFrame, labels_train: pd.DataFrame, labels_test: pd.DataFrame) -> pd.DataFrame:
        # TRAIN SET

        # Compress train data with PCA
        pca = PCA(n_components=self.n_components, svd_solver='full')
        train_data_pca = pca.fit_transform(data_train.to_numpy())

        # Calculate covariance matrix and Mahalanobis distance
        if self.robust_estimator:
            cov = MinCovDet().fit(train_data_pca)
        else:
            cov = EmpiricalCovariance().fit(train_data_pca)
        train_dist = cov.mahalanobis(train_data_pca)

        # Calculate cutoff (anomaly_threshold)
        anomaly_threshold = self.find_anomaly_threshold(
            train_dist, train_data_pca.shape[1])

        # TEST SET

        # Compress test data with PCA
        data_test_pca = pca.transform(data_test.to_numpy())

        # Calculate Mahalanobis distance
        test_dist = cov.mahalanobis(data_test_pca)

        # Detect anomalies
        test_dist = pd.DataFrame(
            test_dist, index=data_test.index, columns=['distance'])
        test_dist.loc[:, Label.ANOMALY] = test_dist.loc[:,
                                                        'distance'] > anomaly_threshold

        # Generate output dataframe
        if self.labels_per_sequence(labels_test):
            # Only labels per SEQUENCE
            output = test_dist.groupby(level=Index.SEQUENCE)[
                Label.ANOMALY].any()
        else:
            # Labels per TIME
            output = test_dist.loc[:, Label.ANOMALY]
        output = pd.DataFrame(output, index=labels_test.index, columns=[
                              Label.ANOMALY])
        return output

    def find_anomaly_threshold(self, values: np.ndarray, n_features: int) -> float:
        if self.threshold_mode == 'chi2':
            return chi2.ppf(self.threshold_value, df=n_features)
        else:
            super().find_anomaly_threshold(values)
