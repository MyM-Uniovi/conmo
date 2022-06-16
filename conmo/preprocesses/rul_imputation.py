import pandas as pd

from conmo.conf import Index, Label
from conmo.preprocesses.preprocess import Preprocess


class RULImputation(Preprocess):

    def __init__(self, threshold: int) -> None:
        self.threshold = threshold

    def apply(self, in_dir, out_dir) -> None:
        """
        RUL imputation per TIME sample based on SEQUENCE labels, generating labels for each TIME.
        """
        self.show_start_message()
        data, labels = self.load_input(in_dir)

        # Add TIME index to labels and set all values to same RUL
        if Index.TIME not in labels.index.names:
            new_labels = pd.DataFrame(index=data.index, columns=labels.columns)
            new_labels.loc[:, Label.RUL] = labels.loc[:, Label.RUL]
            labels = new_labels

        # Add time column and calculate 'max_time' for each sequence
        labels.loc[:, Index.TIME] = labels.index.get_level_values(Index.TIME)
        max_time = labels.groupby(level=[Index.FOLD, Index.SET, Index.SEQUENCE])[Index.TIME].max()

        # Calculate RUL piecewise
        labels.loc[:, Label.RUL] = max_time - \
            labels.loc[:, Index.TIME] + labels.loc[:, Label.RUL]
        labels.clip(upper=self.threshold, inplace=True)

        # Drop unnecessary columns
        labels.drop(columns=Index.TIME, inplace=True)

        self.save_output(out_dir, data, labels)
