from typing import Iterable

import numpy as np
import pandas as pd

from conmo.conf import Index
from conmo.metrics.metric import Metric


class RMSPE(Metric):

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def calculate(self, idx: int, algorithms: Iterable[str], last_preprocess_dir: str, algorithms_dir: str, metrics_dir: str) -> None:
        self.show_start_message()

        # Load ground truth (labels.gz)
        truth = self.load_truth(last_preprocess_dir)
        label = self.problem_label(truth)

        # Create metric dataframe
        folds = truth.index.get_level_values(Index.FOLD).unique()

        # Columns for multiindex
        cols = [algorithms, label]

        # Rows for multiindex
        cycles = [10, 50, 100, 200, 400, 1000]
        rows = [folds, cycles]

        results = pd.DataFrame(data=np.zeros((len(folds)*len(cycles), len(algorithms)*len(label))),
                               index=pd.MultiIndex.from_product(rows), columns=pd.MultiIndex.from_product(cols))

        # Extract test labels
        y_test = truth.loc[1, 'test'].to_numpy()
        # Reshape needed
        y_test = y_test.reshape(-1, 6, y_test.shape[1])

        # Calculate rmspe for each algorithm and fold
        for algorithm in algorithms:
            # Results for each algorithm
            res = self.load_results(algorithm, algorithms_dir)
            # Reshape needed
            res = res.to_numpy().reshape(-1, 6, res.shape[1])

            for fold in folds:
                for cycle in range(6):
                    data = res[:, cycle, :]
                    labels = y_test[:, cycle, :]

                    # LLI results
                    results.loc[fold].loc[:, algorithm].loc[cycles[cycle]
                                                            ].loc[label[0]] = self.rmspe(labels[:, 0], data[:, 0])
                    # LAMPE results
                    results.loc[fold].loc[:, algorithm].loc[cycles[cycle]
                                                            ].loc[label[1]] = self.rmspe(labels[:, 1], data[:, 1])
                    # LAMNE results
                    results.loc[fold].loc[:, algorithm].loc[cycles[cycle]
                                                            ].loc[label[2]] = self.rmspe(labels[:, 2], data[:, 2])

        # Save output
        self.save_output(results, idx, metrics_dir)

    def rmspe(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        '''
        Compute Root Mean Square Percentage Error between two arrays.
        '''
        return np.sqrt(np.mean(np.square((y_true - y_pred)), axis=0))*100
