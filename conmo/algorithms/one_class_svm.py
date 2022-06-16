from typing import Union

import pandas as pd
from sklearn.svm import OneClassSVM as ocsvm

from conmo.conf import Index, Label
from conmo.algorithms.algorithm import AnomalyDetectionClassBasedAlgorithm


class OneClassSVM(AnomalyDetectionClassBasedAlgorithm):

    def __init__(self, kernel: str = 'rbf', degree: int = 3, gamma: Union[str, float] = 'scale', coef0: float = 0.0, tol: float = 1e-3, nu: float = 0.5):
        super().__init__()
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu

    def fit_predict(self, data_train: pd.DataFrame, data_test: pd.DataFrame, labels_train: pd.DataFrame, labels_test: pd.DataFrame) -> pd.DataFrame:
        oc_svm = ocsvm(kernel=self.kernel, degree=self.degree,
                       gamma=self.gamma, coef0=self.coef0, tol=self.tol, nu=self.nu)

        # TRAIN SET

        # Detect the boundary of the train set
        oc_svm = oc_svm.fit(data_train.to_numpy())

        # TEST SET

        # Perform classification on test set
        pred = oc_svm.predict(data_test.to_numpy())

        pred = pd.DataFrame(pred, index=data_test.index,
                            columns=[Label.ANOMALY])
        # Apply mask to One Class SVM result
        # -1 --> outlier (true)
        #  1 --> inlier  (false)
        pred.loc[:, Label.ANOMALY] = pred.loc[:, Label.ANOMALY] == -1

        if self.labels_per_sequence(labels_test):
            # Only labels per SEQUENCE
            output = pred.groupby(level=Index.SEQUENCE)[
                Label.ANOMALY].any()
        else:
            # Labels per TIME
            output = pred.loc[:, Label.ANOMALY]
        output = pd.DataFrame(output, index=labels_test.index, columns=[
                              Label.ANOMALY])
        return output
