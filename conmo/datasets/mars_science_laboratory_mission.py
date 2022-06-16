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


class MarsScienceLaboratoryMission(RemoteDataset):
    URL = 'https://s3-us-west-2.amazonaws.com/telemanom/data.zip'
    FILE_FORMAT = 'zip'
    CHECKSUM = 'c40a236775c1c3a26de601f66541b414'
    CHECKSUM_FORMAT = 'md5'
    VARIABLES = ['dim_{:02}'.format(x) for x in range(1, 56)]
    LABEL = 'anomaly'
    CHANNELS = ['M-06', 'M-01', 'M-02', 'S-02', 'P-10', 'T-04', 'T-05', 'F-07', 'M-03', 'M-04', 'M-05', 'P-15', 'C-01', 'C-02', 'T-12', 'T-13', 'F-04', 'F-05', 'D-14', 'T-09', 'P-14', 'T-08',
                'P-11', 'D-15', 'D-16', 'M-07', 'F-08']
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
        Method in charge of downloading and parsing the MLS dataset labels files. 
        This is because the tags are located at a different URL than the data.

        Parameters
        ----------
        raw_dir: str
            Directory were the unparsed data of SMAP dataset is stored until it's processed.

        Returns
        -------
        labeled_anomalies: Pandas Dataframe
            Anomalous intervals in the MSL dataset.
        """
        # Download from GitHub repository
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

        # Only MSL anomalies
        labeled_anomalies = labeled_anomalies.loc[
            labeled_anomalies['spacecraft'] == 'MSL']

        return labeled_anomalies

    def check_checksum_lbl(self, response: object, checksum: str) -> bool:
        """
        Checks if the checksum of the downloaded file corresponds to the one provided in the class.
        For security e integrity issues. Currently only the md5 algorithm is integrated.
        Since in the MLS dataset the labels are obtained from a different file, it's necessary to use another method 
        to pass the checksum of that file.

        Parameters
        ----------
        response: Object
            Response object returned by the get method of the Requests library.
        checksum: str
            String containing the labels checksum.

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
            Channel identifier (subdataset).
        labeled_anormalies: Pandas Dataframe
            Anomalous intervals in the MSL dataset.

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
        MSL dataset has only 2 sequences: one for train and another for test.

        Return
        ------
        array
            List with the index for each sequence of the dataset.
        """
        return [-1, 0]
