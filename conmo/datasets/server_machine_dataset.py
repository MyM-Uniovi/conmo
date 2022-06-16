import os
import shutil
from os import path
from typing import Iterable

import numpy as np
import pandas as pd

from conmo.conf import File, Index, Label
from conmo.datasets.dataset import RemoteDataset


class ServerMachineDataset(RemoteDataset):
    URL = 'https://github.com/NetManAIOps/OmniAnomaly/archive/refs/heads/master.zip'
    FILE_FORMAT = 'zip'
    CHECKSUM = 'ebef7f89e73cdc9e926bdbf8bb591447'
    CHECKSUM_FORMAT = 'md5'
    VARIABLES = ['cpu_r', 'load_1', 'load_5', 'load_15', 'mem_shmem', 'mem_u', 'mem_u_e', 'total_mem', 'disk_q', 'disk_r', 'disk_rb', 'disk_svc', 'disk_u', 'disk_w', 'disk_wa', 'disk_wb', 'si', 'so', 'eth1_fi', 'eth1_fo', 'eth1_pi', 'eth1_po',
                 'tcp_tw', 'tcp_use', 'active_opens', 'curr_estab', 'in_errs', 'in_segs', 'listen_overflows', 'out_rsts', 'out_segs', 'passive_opens', 'retransegs', 'tcp_timeouts', 'udp_in_dg', 'udp_out_dg', 'udp_rcv_buf_errs', 'udp_snd_buf_errs']
    LABEL = Label.ANOMALY
    SUBDATASETS = ['1-01', '1-02', '1-03', '1-04', '1-05', '1-06', '1-07', '1-08', '2-01', '2-02', '2-03', '2-04', '2-05', '2-06', '2-07', '2-08' , '2-09', '3-01', '3-02', '3-03', '3-04', '3-05', '3-06', '3-07', '3-08', '3-09', '3-10', '3-11']
    SEQUENCE_COLUMN = 'machine'
    TIME_COLUMN = 'time'

    def __init__(self, subdataset: str) -> None:
        super().__init__(self.URL, self.FILE_FORMAT, self.CHECKSUM, self.CHECKSUM_FORMAT)
        if subdataset not in self.SUBDATASETS:
            raise RuntimeError("Invalid selected subdataset")
        self.subdataset = subdataset

    def dataset_files(self) -> Iterable:
        files = []
        for key in self.SUBDATASETS:
            files.append(path.join(self.dataset_dir,
                         "{}_{}".format(key, File.DATA)))
            files.append(path.join(self.dataset_dir,
                         "{}_{}".format(key, File.LABELS)))
        return files

    def parse_to_package(self, raw_dir: str) -> None:
        # Warning: For SMD dataset, it's downloaded from the entire Github repository so the dataset is located in a subfolder
        raw_dir = path.join(raw_dir, 'OmniAnomaly-master',
                            'ServerMachineDataset')

        for subdataset in os.listdir(path.join(raw_dir, 'train')):
            # Read TRAIN and TEST DATA and generate dataframe
            train = pd.read_csv(path.join(raw_dir, 'train', subdataset),
                                sep=',', header=None, names=self.VARIABLES)
            # Reset index for starting from 1 (TIME)
            train.index += 1

            test = pd.read_csv(path.join(raw_dir, 'test', subdataset),
                               sep=',', header=None, names=self.VARIABLES)
            # Reset index for starting from 1 (TIME)
            test.index += 1

            # Generate DATA dataframe
            data = pd.concat([train, test], keys=[1, 2], names=[
                Index.SEQUENCE, Index.TIME])
            data.sort_index(inplace=True)

            # Generate LABELS dataframe and fill it with train anomaly labels set to 0
            labels = pd.read_csv(path.join(raw_dir, 'test_label', subdataset),
                                 sep=',', header=None, names=[self.LABEL])

            # Reset index for starting from 1 (TIME)
            labels.index += 1

            # Reindex labels with test/train index and fill it with train anomaly labels set to 0
            labels = pd.concat([labels], keys=[1], names=[
                Index.SEQUENCE, Index.TIME])
            labels = labels.reindex(data.index, fill_value=0)
            labels.sort_index(inplace=True)

            # Extract file data (for identifying subdatasets)
            group_index = subdataset.split('-')[1]
            index = subdataset.split('-')[2].split('.')[0]

            # Save dataframes
            data.to_parquet(path.join(self.dataset_dir, "{:01}-{:02}_{}".format(
                int(group_index), int(index), File.DATA)), compression="gzip", index=True)
            labels.to_parquet(path.join(self.dataset_dir, "{:01}-{:02}_{}".format(
                int(group_index), int(index), File.LABELS)), compression="gzip", index=True)

    def feed_pipeline(self, out_dir: str) -> None:
        shutil.copy(path.join(self.dataset_dir, "{}_{}".format(
            self.subdataset, File.DATA)), path.join(out_dir, File.DATA))
        shutil.copy(path.join(self.dataset_dir, "{}_{}".format(
            self.subdataset, File.LABELS)), path.join(out_dir, File.LABELS))

    def sklearn_predefined_split(self) -> Iterable[int]:
        """
        Generates array of indexes of same length as sequences to be used with 'PredefinedSplit'
        SMD dataset has only 2 sequences: one for train and another for test.

        Returns
        -------
        array
            List with the index for each sequence of the dataset.
        """
        return [-1, 0]
