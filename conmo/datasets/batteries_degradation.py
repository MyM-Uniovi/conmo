import os
import shutil
from os import path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import PchipInterpolator as pchip

from conmo.conf import File, Index
from conmo.datasets.dataset import LocalDataset


class BatteriesDataset(LocalDataset):
    """
    This is a dataset obtained from measurements of certain types of degradation of three types of batteries.
    Since it belongs to the local datasets, to launch any experiment with it, it must be stored on disk with 
    the following directory structure:
    - DTW-Li-ion-Diagnosis
        - data : Data and labels for the three types of batteries 
                    are stored here.
        - mat: 
            - LFP:
                - diagnosis:
                    - V.mat
                - test:
                    - V_references.mat
                    - x_test_0.mat
                    - x_test_1.mat
                    - x_test_2.mat
                    - x_test_3.mat
                    - y_test.mat
            - NCA:
                - diagnosis
                - test
            - NMC:
                - The same as NCA and LFP
            - Q.mat
    """

    CHEMISTRY_LIST = ['LFP', 'NCA', 'NMC']
    MIN_V = 0
    MAX_V = 0
    SIZE = 128
    UI_STEP = 0.0005
    MIN_V_LFP = 3.20
    MAX_V_LFP = 3.50
    MIN_V_NCA = 3.20
    MAX_V_NCA = 4.23
    MIN_V_NMC = 3.44
    MAX_V_NMC = 4.28

    def __init__(self, path: str, chemistry: str, test_set: int) -> None:
        super().__init__(path)
        if chemistry not in self.CHEMISTRY_LIST:
            raise RuntimeError("Invalid selected chemistry")
        if test_set not in range(4):
            raise RuntimeError("Invalid selected testing data")
        self.path = path
        self.test_set = test_set
        self.chemistry = chemistry
        self.MIN_V, self.MAX_V, _ = self.get_minmaxV(chemistry)

    def dataset_files(self) -> Iterable:
        files = []
        for chemistry in self.CHEMISTRY_LIST:
            for test_idx in range(4):
                files.append(path.join(self.dataset_dir,
                                       "{}-{:02}_{}".format(chemistry, test_idx, File.DATA)))
                files.append(path.join(self.dataset_dir,
                                       "{}-{:02}_{}".format(chemistry, test_idx, File.LABELS)))
        return files

    def load(self) -> None:
        """
        Parse dataset train/test data to match Conmo's standard.
        """
        path_data = path.join(self.path, 'data')
        path_mat = path.join(self.path, 'mat')

        for chemistry in self.CHEMISTRY_LIST:
            # Read TRAIN DATA and generate dataframe
            train_data_np = np.load(
                path.join(path_data, 'x_train_' + chemistry + '.npy'))
            train_data = pd.DataFrame(train_data_np, columns=[
                                      "feature_{:03}".format(i) for i in range(127)])
            # Reset index for starting from 1
            train_data.index += 1

            # Read TRAIN LABELS and generate dataframe
            train_labels_np = np.load(
                path.join(path_data, 'y_train_' + chemistry + '.npy'))
            train_labels = pd.DataFrame(train_labels_np, columns=[
                                        'LLI', 'LAMPE', 'LAMNE'])
            # Reset index for starting from 1
            train_labels.index += 1

            # Load capacity file (needed later)
            Q = sio.loadmat(path.join(path_mat, 'Q.mat'))['Qnorm'].flatten()

            # Load TEST LABELS (the same over all types of test data)
            test_labels_np = sio.loadmat(
                path.join(path_mat, chemistry, 'test', 'y_test.mat'))['y_test']

            # Reshape labels from (num_samples, cycles, sample_size) to (num_samples*cycles, degradation_modes)
            test_labels_np = test_labels_np / 100
            test_labels_np = test_labels_np.reshape(-1,
                                                    test_labels_np.shape[2])
            test_labels = pd.DataFrame(test_labels_np, columns=[
                                       'LLI', 'LAMPE', 'LAMNE', 'capacity_loss'])

            # Delete last feature (capacity_loss) unusued in this problem
            test_labels.drop('capacity_loss', axis=1, inplace=True)

            # Reset index for starting from 1
            test_labels.index += 1

            # Iterate over different types of test data degradation
            for idx in range(4):
                # Read TEST DATA and generate dataframe
                test_data_np = sio.loadmat(
                    path.join(path_mat, chemistry, 'test', 'x_test_{}.mat'.format(idx)))['x_test'].T
                # (n_samples, seq_len)
                test_data_np = test_data_np.reshape(-1, test_data_np.shape[2])
                test_data_np = self.convert_to_input_data(
                    test_data_np, Q, self.SIZE-1, chemistry)

                test_data_np = self.normalise_data(
                    test_data_np, np.min(train_data_np), np.max(train_data_np))

                # Convert to Pandas dataframe
                test_data = pd.DataFrame(test_data_np, columns=[
                                         "feature_{:03}".format(i) for i in range(127)])
                # Reset index for starting from 1
                test_data.index += 1

                # Generate DATA dataframe
                data = pd.concat([train_data, test_data], keys=[
                                 1, 2], names=[Index.SEQUENCE, Index.TIME])
                data.sort_index(inplace=True)

                # Generate LABELS dataframe
                labels = pd.concat([train_labels, test_labels], keys=[
                                   1, 2], names=[Index.SEQUENCE, Index.TIME])
                labels.sort_index(inplace=True)

                # Save parsed dataframes to disk
                data.to_parquet(path.join(self.dataset_dir, "{}-{:02}_{}".format(
                    chemistry, idx, File.DATA)), compression="gzip", index=True)
                labels.to_parquet(path.join(self.dataset_dir, "{}-{:02}_{}".format(
                    chemistry, idx, File.LABELS)), compression="gzip", index=True)

    def feed_pipeline(self, out_dir: str) -> None:
        """
        Copy selected data file to pipeline step folder.

        Parameters
        ----------
        out_dir:
            Directory where the dataset was originally stored.
        """
        shutil.copy(path.join(self.dataset_dir, "{}-{:02}_{}".format(
            self.chemistry, self.test_set, File.DATA)), path.join(out_dir, File.DATA))
        shutil.copy(path.join(self.dataset_dir, "{}-{:02}_{}".format(
            self.chemistry, self.test_set, File.LABELS)), path.join(out_dir, File.LABELS))

    def sklearn_predefined_split(self) -> Iterable[int]:
        """
        Generates array of indexes of same length as sequences to be used with 'PredefinedSplit'

        Returns
        -------
        array, list with the index for each sequence of the dataset.
        """
        return [-1, 0]

    def convert_to_input_data(self, ui_new: list, Q: list, size: int, material: int) -> np.ndarray:
        '''
        Converts the voltage values of the real cells to the input data for the neural network

        Parameters
        ----------
        ui_new: array
            Voltage values of the cell at each cycle in percentage.
        Q: array
            Capacity percentages from 0 to 100 from the simulated dataset.
        size: int
            The length of the curves.
        material: str
            Chemistry of the cell.

        Returns
        -------
        x_test: array
            The input data for the neural network.
        '''
        min_v, max_v, = self.MIN_V, self.MAX_V
        samples = []
        for sample in range(len(ui_new)):
            # convert to IC
            ui_sample, dqi_sample = self.IC(
                ui_new[sample], Q, self.UI_STEP, min_v, max_v)
            # reduce size
            new_sample = self.reduce_size(ui_sample, dqi_sample, size)
            samples.append(new_sample)
        x_test = np.array(samples)
        return x_test

    def IC(self, u: np.ndarray, q: np.ndarray, ui_step: float = 0.0005, minV: float = 3.2, maxV: float = 3.5) -> (np.ndarray, np.ndarray):
        '''
        Get the ICA data for a given voltage curve

        Parameters
        ----------
        u: numpy array
            Voltage curve.
        q: numpy array
            Capacity curve.
        ui_step: float
            Step of interpolation.
        minV: float
            Minimum voltage of the IC curve.
        maxV: float
            Maximum voltage of the IC curve.

        Returns
        -------
        ui, dqi: numpy arrays
            Interpolated voltage and derivative of capacity
        '''

        # voltages values for which capacity is interpolated
        ui = np.arange(minV, maxV, ui_step)
        qi = np.interp(ui, u, q)
        return ui[1:], np.diff(qi)

    def reduce_size(self, ui: np.ndarray, dqi: np.ndarray, size: int) -> np.ndarray:
        '''
        Reduces the length of the IC data to a given size

        Parameters
        ----------
        ui: numpy array
            Voltage curve.
        dqi: numpy array
            Derivative of capacity (IC).
        size: int 
            Size at which to reduce the IC data.

        Returns
        -------
        numpy array
            Reduced IC.
        '''

        curve = pchip(ui, dqi)
        ui_reduced = np.linspace(min(ui), max(ui), size)
        return curve(ui_reduced)

    def normalise_data(self, data: np.ndarray, min_val: float, max_val: float, low: int = 0, high: int = 1) -> float:
        '''
        Normalises the data to the range [low, high]

        Parameters
        ----------
        data: numpy array
            Data to normalise.
        min: float
            Minimum value of data.
        max: float
            Maximum value of data.
        low: float
            Minimum value of the range.
        high: float
            Maximum value of the range.

        Returns
        -------
        normalised_data: float
            normalised data
        '''
        normalised_data = (data - min_val)/(max_val - min_val)
        normalised_data = (high - low)*normalised_data + low
        return normalised_data

    def get_minmaxV(self, material: np.ndarray) -> (int, int, str):
        '''
        Returns the range voltage in which to study the IC curves

        Parameters
        ----------
        material: numpy array
            Chemistry to study.

        Returns
        -------
        min_v, max_v, path: numpy arrays, str
            Min and max voltage values and path where data is located,
        '''
        min_v = -1
        max_v = -1
        tmp_path = path.join(self.path, 'mat', material, 'diagnosis')
        if material == "LFP":
            min_v = self.MIN_V_LFP
            max_v = self.MAX_V_LFP
        elif material == "NCA":
            min_v = self.MIN_V_NCA
            max_v = self.MAX_V_NCA
        elif material == "NMC":
            min_v = self.MIN_V_NMC
            max_v = self.MAX_V_NMC
        else:
            print("ERROR: Chemistry not found")
            return -1
        if min_v == -1 or max_v == -1 or path == "":
            print("ERROR: Chemistry not found")
            return -1
        return min_v, max_v, tmp_path
