import shutil
from os import path
from typing import Iterable

import numpy as np
import pandas as pd

from conmo.conf import File, Index, Label
from conmo.datasets.dataset import RemoteDataset


class NASATurbofanDegradation(RemoteDataset):
    URL = "https://ti.arc.nasa.gov/c/6/"
    FILE_FORMAT = "zip"
    CHECKSUM = "79a22f36e80606c69d0e9e4da5bb2b7a"
    CHECKSUM_FORMAT = "md5"
    SUBDATASETS = {
        "FD001": {
            "train": 100,
            "test": 100
        },
        "FD002": {
            "train": 260,
            "test": 259
        },
        "FD003": {
            "train": 100,
            "test": 100
        },
        "FD004": {
            "train": 249,
            "test": 248
        }
    }
    VARIABLES = ["setting_1", "setting_2", 'TRA', 'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc',
                 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
    LABEL = Label.RUL
    SEQUENCE_COLUMN = 'unit_number'
    TIME_COLUMN = 'time_cycles'

    def __init__(self, subdataset: str) -> None:
        super().__init__(self.URL, self.FILE_FORMAT, self.CHECKSUM, self.CHECKSUM_FORMAT)
        if subdataset not in self.SUBDATASETS:
            raise RuntimeError("Invalid selected subdataset")
        self.subdataset = subdataset

    def dataset_files(self) -> Iterable:
        files = []
        for key in self.SUBDATASETS.keys():
            files.append(path.join(self.dataset_dir,
                         "{}_{}".format(key, File.DATA)))
            files.append(path.join(self.dataset_dir,
                         "{}_{}".format(key, File.LABELS)))
        return files

    def parse_to_package(self, raw_dir: str) -> None:
        columns = [self.SEQUENCE_COLUMN]
        columns.append(self.TIME_COLUMN)
        columns.extend(self.VARIABLES)

        for subdataset in self.SUBDATASETS:
            # Read raw files
            train = pd.read_csv(path.join(raw_dir, "train_" + subdataset + ".txt"),
                                sep='\s+', header=None, names=columns)
            test = pd.read_csv(path.join(raw_dir, "test_" + subdataset + ".txt"),
                               sep='\s+', header=None, names=columns)
            rul_test = pd.read_csv(path.join(raw_dir, "RUL_" + subdataset + ".txt"),
                                   sep='\s+', header=None, names=[self.LABEL])

            # Modify unit_number for test subset before merging with train
            test.loc[:, self.SEQUENCE_COLUMN] += self.SUBDATASETS[subdataset]['train']

            # Generate dataframe with multiindex SEQUENCE > TIME
            data = pd.concat([train, test], ignore_index=True)
            data.set_index(
                [self.SEQUENCE_COLUMN, self.TIME_COLUMN], inplace=True)
            data.rename_axis(index={self.SEQUENCE_COLUMN: Index.SEQUENCE,
                                    self.TIME_COLUMN: Index.TIME}, inplace=True)
            data.sort_index(inplace=True)

            # Generate labels according to data indexes
            labels = pd.DataFrame(index=data.index.unique(
                level=Index.SEQUENCE), columns=[self.LABEL])
            labels.loc[:self.SUBDATASETS[subdataset]['train'], self.LABEL] = 0
            labels.loc[self.SUBDATASETS[subdataset]['train']+1:,
                       self.LABEL] = rul_test.loc[:, self.LABEL].to_numpy()

            # Save dataframes
            data.to_parquet(path.join(self.dataset_dir, "{}_{}".format(
                subdataset, File.DATA)), compression="gzip", index=True)
            labels.to_parquet(path.join(self.dataset_dir, "{}_{}".format(
                subdataset, File.LABELS)), compression="gzip", index=True)

    def feed_pipeline(self, out_dir: str) -> None:
        shutil.copy(path.join(self.dataset_dir, "{}_{}".format(
            self.subdataset, File.DATA)), path.join(out_dir, File.DATA))
        shutil.copy(path.join(self.dataset_dir, "{}_{}".format(
            self.subdataset, File.LABELS)), path.join(out_dir, File.LABELS))

    def sklearn_predefined_split(self) -> Iterable[int]:
        """
        Generates array of indexes of same length as sequences to be used with 'PredefinedSplit'
        
        Returns
        -------
        array 
            List with the index for each sequence of the dataset.
        """
        # Generate array of indexes of same length as sequences to be used with 'PredefinedSplit' of 'scikit-learn'
        idx = np.empty(self.SUBDATASETS[self.subdataset]['train'] +
                       self.SUBDATASETS[self.subdataset]['test'], dtype=int)

        # Set first sequences as train and latest to test (order set when generating DataFrame)
        idx[:self.SUBDATASETS[self.subdataset]['train']] = -1
        idx[-self.SUBDATASETS[self.subdataset]['test']:] = 0

        return idx
