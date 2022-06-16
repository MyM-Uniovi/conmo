from os import path

import numpy as np
import pandas as pd
import pytest

from conmo.conf import File, Testing

@pytest.fixture(scope="package")
def create_synthetic_dataset(tmp_path_factory) -> str:
    """
    Create a synthetic dataset for testing purposes (Algorithms)
    Like the datasets included in the framework, this dataset contains a multivariate time series splittered in 2 folds with:
        - 1 column index for fold
        - 1 column index for train/test set
        - 1 column index for the sequences
        - 1 column index for the time
        - 6 columns for attributes with random values between 1 and 20

    The labels contain:
        - 1 column index for fold
        - 1 column for sequence
        - 1 column for time
        - 1 column with the labels ('1' if there is an outlier or '0' if the sample is an inlier)
    """
    # Generate session-temporary data directory to save the synthetic dataset
    dataset_dir = tmp_path_factory.mktemp('data')

    # Generate train indexes
    train_time = [i for i in range(1,501)]
    train_sequence = [1 for _ in range(1, 501)]
    train_set = ['train' for _ in range(1,501)]

    # Generate test indexes
    test_time = [i for i in range(1, 251)]
    test_sequence = [2 for _ in range(1,251)]
    test_set = ['test' for _ in range(1,251)]

    # Create multiindex
    idx = [train_set + test_set, train_sequence + test_sequence, train_time + test_time]
    tuples = list(zip(*idx))
    multiindex = pd.MultiIndex.from_tuples(tuples, names=['set', 'sequence', 'time'])

    # Random integer generator
    rng = np.random.default_rng(seed=Testing.RANDOM_SEED)

    # Create data folds 
    data_fold_1 = pd.DataFrame(rng.integers(1, 20, size=(750, 6)), 
                columns=['dim_01', 'dim_02', 'dim_03', 'dim_04', 'dim_05', 'dim_06'], index=multiindex)
    data_fold_2 = pd.DataFrame(rng.integers(1, 20, size=(750,6)),
                columns=['dim_01', 'dim_02', 'dim_03', 'dim_04', 'dim_05', 'dim_06'], index=multiindex)

    data = pd.concat([data_fold_1, data_fold_2], keys=[1, 2], names=['fold'])
    
    # Create labels folds
    lbl_fold_1 = pd.DataFrame(rng.integers(0, 1, size=(750, 1)),
                                columns=['anomaly'], index=multiindex)
    lbl_fold_2 = pd.DataFrame(rng.integers(0, 1, size=(750, 1)),
                                columns=['anomaly'], index=multiindex)

    labels = pd.concat([lbl_fold_1, lbl_fold_2], keys=[1, 2], names=['fold'])

    # Save dataframes to temporal directory
    data.to_parquet(path.join(dataset_dir, File.DATA), compression="gzip", index=True)
    labels.to_parquet(path.join(dataset_dir, File.LABELS), compression="gzip", index=True)
    
    return dataset_dir

