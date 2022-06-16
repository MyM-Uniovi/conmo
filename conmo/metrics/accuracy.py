from typing import Iterable

import pandas as pd
from sklearn.metrics import accuracy_score

from conmo.conf import Index
from conmo.metrics.metric import Metric


class Accuracy(Metric):

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def calculate(self, idx: int, algorithms: Iterable[str], last_preprocess_dir: str, algorithms_dir: str, metrics_dir: str) -> None:
        self.show_start_message()

        # Load ground truth
        truth = self.load_truth(last_preprocess_dir)
        label = self.problem_label(truth)

        # Create metric dataframe
        folds = truth.index.get_level_values(Index.FOLD).unique()
        acc = pd.DataFrame(index=folds, columns=algorithms)

        # Calculate accuracy for each algorithm and fold
        for algorithm in algorithms:
            res = self.load_results(algorithm, algorithms_dir)
            for fold in folds:
                acc.loc[fold, algorithm] = accuracy_score(
                    truth.loc[(fold, Index.SET_TEST), label].values, res.loc[fold, label].values, normalize=self.normalize)

        # Save output
        self.save_output(acc, idx, metrics_dir)
