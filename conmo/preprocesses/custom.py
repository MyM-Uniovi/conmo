from typing import Callable, Tuple

import pandas as pd

from conmo.preprocesses.preprocess import Preprocess


class CustomPreprocess(Preprocess):
    """
    Core class used to implement self-created preprocess.
    Such preprocess will be wrapped in a function that will be passed as an argument to the constructor of this class.
    """

    def __init__(self, fn: Callable[[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]) -> None:
        self.fn = fn

    def apply(self, in_dir: str, out_dir: str) -> None:
        """
        Applies the custom preprocess to labels and data.

        Parameters
        ----------
        in_dir: str
            Input directory where the files are located. Usually, this is the output directory of the splitter step.
        out_dir: str
            Output directory where the files will be saved.
        """
        self.show_start_message()
        data, labels = self.load_input(in_dir)

        # Apply custom preprocess
        data, labels = self.fn(data, labels)

        self.save_output(out_dir, data, labels)