import ast
import hashlib
import shutil
from os import path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from conmo.conf import File, Index
from conmo.datasets.dataset import RemoteDataset


class SoilMoistureActivePassiveSatellite(RemoteDataset):
    URL = 'https://s3-us-west-2.amazonaws.com/telemanom/data.zip'
    FILE_FORMAT = 'zip'
    CHECKSUM = 'c40a236775c1c3a26de601f66541b414'
    CHECKSUM_FORMAT = 'md5'
    VARIABLES = ['dim_01', 'dim_02', 'dim_03', 'dim_04', 'dim_05', 'dim_06', 'dim_07', 'dim_08', 'dim_09', 'dim_10', 'dim_11', 'dim_12', 'dim_13', 'dim_14', 'dim_15', 'dim_16', 'dim_17',
                 'dim_18', 'dim_19', 'dim_20', 'dim_21', 'dim_22', 'dim_23', 'dim_24', 'dim_25']
    LABEL = 'anomaly'
    CHANNELS = ['P-01', 'S-01', 'E-01', 'E-02', 'E-03', 'E-04', 'E-05', 'E-06', 'E-07', 'E-08', 'E-09', 'E-10', 'E-11', 'E-12', 'E-13', 'A-01', 'D-01', 'P-02', 'P-03', 'D-02', 'D-03', 'D-04', 'A-02', 'A-03', 'A-04', 'G-01', 'G-02', 'D-05',
                'D-06', 'D-07', 'F-01', 'P-04', 'G-03', 'T-01', 'T-02', 'D-08', 'D-09', 'F-02', 'G-04', 'T-03', 'D-11', 'D-12', 'B-01', 'G-06', 'G-07', 'P-07', 'R-01', 'A-05', 'A-06', 'A-07', 'D-13', 'P-02', 'A-08', 'A-09', 'F-03']
    SEQUENCE_COLUMN = 'machine'
    TIME_COLUMN = 'time'

    def __init__(self, channel: str) -> None:
        super().__init__(self.URL, self.FILE_FORMAT, self.CHECKSUM, self.CHECKSUM_FORMAT)
        if channel not in self.CHANNELS:
            raise RuntimeError("Invalid selected subdataset")
        self.channel = channel

    def dataset_files(self) -> Iterable:
        files = []
        for key in self.CHANNELS:
            files.append(path.join(self.dataset_dir,
                                   "{}_{}".format(key, File.DATA)))
            files.append(path.join(self.dataset_dir,
                                   "{}_{}".format(key, File.LABELS)))
        return files

    def parse_to_package(self, raw_dir: str) -> None:
        raw_dir = path.join(raw_dir, 'data')

        # Obtain dataset with anomalies
        labeled_anomalies = self.download_anomalies_file(raw_dir)

        for channel in self.CHANNELS:
            # Recreate the name of the npy file
            ch_name = '{}-{:01}'.format(channel.split('-')[0],
                                        int(channel.split('-')[1]))
            file_name = ch_name + '.npy'

            # Read TRAIN and TEST DATA and generate dataframe
            train = np.load(path.join(raw_dir, 'train', file_name))
            train = pd.DataFrame.from_records(train, columns=self.VARIABLES)
            # Reset index for starting from 1 (TIME)
            train.index += 1

            test = np.load(path.join(raw_dir, 'test', file_name))
            test = pd.DataFrame.from_records(test, columns=self.VARIABLES)
            # Reset index for starting from 1 (TIME)
            test.index += 1

            # Generate DATA dataframe
            data = pd.concat([train, test], keys=[1, 2], names=[
                Index.SEQUENCE, Index.TIME])
            data.sort_index(inplace=True)

            # Generate TRAIN sand TEST LABELS dataframe and fill it with zeros
            test_labels = pd.DataFrame(
                np.zeros(test.shape[0]), columns=[self.LABEL])
            train_labels = pd.DataFrame(
                np.zeros(train.shape[0]),  columns=[self.LABEL])

            # Reset index for starting from 1 in both label's datasets
            train_labels.index += 1
            test_labels.index += 1

            # Represent anomalies in the label's dataset following 'labeled_anomalies.csv'
            test_labels = self.represent_anomalies(
                test_labels, ch_name, labeled_anomalies)

            # Reindex labels with test/train index and fill it with train anomaly labels set to 0
            labels = pd.concat([test_labels, train_labels], keys=[1], names=[
                Index.SEQUENCE, Index.TIME])
            labels = labels.reindex(data.index, fill_value=0)
            labels.sort_index(inplace=True)

            # Save dataframes
            data.to_parquet(path.join(self.dataset_dir, "{}_{}".format(
                channel, File.DATA)), compression="gzip", index=True)
            labels.to_parquet(path.join(self.dataset_dir, "{}_{}".format(
                channel, File.LABELS)), compression="gzip", index=True)

    def feed_pipeline(self, out_dir: str) -> None:
        shutil.copy(path.join(self.dataset_dir, "{}_{}".format(
            self.channel, File.DATA)), path.join(out_dir, File.DATA))
        shutil.copy(path.join(self.dataset_dir, "{}_{}".format(
            self.channel, File.LABELS)), path.join(out_dir, File.LABELS))

    def download_anomalies_file(self, raw_dir: str) -> Iterable[pd.DataFrame]:
        """
        Method in charge of downloading and parsing the SMAP dataset labels files. 
        This is because the tags are located at a different URL than the data.

        Parameters
        ----------
        raw_dir: str
            Directory were the unparsed data of SMAP dataset is stored until it's processed.

        Returns
        -------
        labeled_anomalies: Pandas Dataframe
            Anomalous intervals in the SMAP dataset.
        """
        # Download fro GitHub repository
        url = 'https://github.com/khundman/telemanom/archive/refs/heads/master.zip'
        checksum = '5ffd3202e3d1827f8caaf89ec6c14a88'
        r = requests.get(url, stream=True)
        if not r.ok:
            raise ConnectionError(
                "An error occurred downloading {}.".format(self.name))
        if not self.check_checksum_lbl(r, checksum):
            raise RuntimeError(
                "{} has a checksum differing from expected, file may be corrupted.").format(self.name)
        # Extract data from ZIP file
        self.extract_data(r, raw_dir)

        # Read CSV with anomalies periods
        labeled_anomalies = pd.read_csv(path.join(raw_dir, 'telemanom-master', 'labeled_anomalies.csv'),
                                        sep=',', usecols=['chan_id', 'spacecraft', 'anomaly_sequences'])

        # Only SMAP anomalies
        labeled_anomalies = labeled_anomalies.loc[
            labeled_anomalies['spacecraft'] == 'SMAP']

        return labeled_anomalies

    def check_checksum_lbl(self, response: object, checksum: str) -> bool:
        """
        Checks if the checksum of the downloaded file corresponds to the one provided in the class.
        For security e integrity issues. Currently only the md5 algorithm is integrated.
        Since in the SMAP dataset the labels are obtained from a different file, it's necessary to use another method 
        to pass the checksum of that file.

        Parameters
        ----------
        response: object
            Response object returned by the get method of the Requests library.
        checksum: str
            String containing the labels' checksum.

        Returns
        -------
        bool
            Boolean variable indicating whether the comparison of the hash with the checksum was successful or not.
        """
        if self.checksum_format == 'md5':
            # MD5 checksum
            md5 = hashlib.md5(response.content).hexdigest()
            if md5 == checksum:
                return True
            else:
                return False

    def represent_anomalies(self, labels: Iterable[pd.DataFrame], channel: str, labeled_anomalies: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
        """
        Represent anomalies in the label's dataset following the anomalous intervals of 'labeled_anomalies.csv'

        Parameters
        ----------
        labels: Pandas Dataframe
            Dataframe with the shape of the labels but filled wth zeros.
        channel: str
            Channel identifier (subdataset)
        labeled_anormalies: Pandas Dataframe 
            Anomalous intervals in the SMAP dataset.

        Returns
        -------
        labels: Pandas Dataframe
            Labels dataset correctly filled.
        """
        # Obtain anomaly sequences of the given channel and transform them into a list
        anomaly_sequences = (labeled_anomalies.loc[
            labeled_anomalies['chan_id'] == channel]).iloc[0, 2]

        # Convert from string to tuple
        anomaly_sequences = ast.literal_eval(anomaly_sequences)

        # Iterate over the sequences list of anomalies
        for seq in anomaly_sequences:
            labels.iloc[seq[0]:seq[1]] = 1

        return labels

    def sklearn_predefined_split(self) -> Iterable[int]:
        """
        Generates array of indexes of same length as sequences to be used with 'PredefinedSplit'
        SMAP dataset has only 2 sequences: one for train and another for test.

        Returns
        -------
        array
            List with the index for each sequence of the dataset.
        """
        return [-1, 0]
