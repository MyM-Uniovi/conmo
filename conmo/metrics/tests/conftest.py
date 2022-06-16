from os import path

import numpy as np
import pandas as pd
import pytest

from conmo.conf import File, Testing

@pytest.fixture(scope="package")
def create_synthetic_ground_truth(tmp_path_factory) -> str:
    """
    Create synthetic data for recreating the ground truth of a dataset (testing purposes)
    """
    # Generate session-temporary data directory to save the synthetic ground truth
    ground_truth_dir = tmp_path_factory.mktemp('ground_truth')

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

    # Create labels folds
    lbl_fold_1 = pd.DataFrame(rng.integers(0, 1, size=(750, 1)),
                                columns=['anomaly'], index=multiindex)
    lbl_fold_2 = pd.DataFrame(rng.integers(0, 1, size=(750, 1)),
                                columns=['anomaly'], index=multiindex)

    labels = pd.concat([lbl_fold_1, lbl_fold_2], keys=[1, 2], names=['fold'])

    # Save dataframe to temporal directory
    labels.to_parquet(path.join(ground_truth_dir, File.LABELS), compression="gzip", index=True)
    
    return ground_truth_dir

@pytest.fixture(scope="package")
def create_synthetic_results(tmp_path_factory) -> str:
    """
    Create synthetic data for recreating the results of an execution of an algorithm over a dataset (testing purposes)
    """
    # Generate session-temporary data directory to save the synthetic results of an algorithm
    results_dir = tmp_path_factory.mktemp('results')

    # Generate indexes indexes
    results_time = [i for i in range(1,251)]
    results_sequence = [1 for _ in range(1, 251)]

    # Create multiindex
    idx = [results_sequence, results_time]
    tuples = list(zip(*idx))
    multiindex = pd.MultiIndex.from_tuples(tuples, names=['sequence', 'time'])

    # Random integer generator
    rng = np.random.default_rng(seed=Testing.RANDOM_SEED)
   
    # Create results folds
    results_fold_1 = pd.DataFrame(rng.integers(0, 1, size=(250, 1)),
                                columns=['anomaly'], index=multiindex)
    results_fold_2 = pd.DataFrame(rng.integers(0, 1, size=(250, 1)),
                                columns=['anomaly'], index=multiindex)

    labels = pd.concat([results_fold_1, results_fold_2], keys=[1, 2], names=['fold'])

    # Save dataframe to temporal directory
    labels.to_parquet(path.join(results_dir, '01_KerasAutoencoder.gz'), compression="gzip", index=True)
    
    return results_dir