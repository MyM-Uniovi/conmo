from os import path

import numpy as np
import pandas as pd

from conmo.algorithms import OneClassSVM

OUTPUT_INDEX = ['fold', 'sequence', 'time']
OUTPUT_COLUMNS = ['anomaly']

def test_one_class_svm(tmp_path_factory, create_synthetic_dataset):
    # Generate temporary output directory for feeding pipeline 
    out_dir = tmp_path_factory.mktemp('output')

    # Instantiate and execute algorithm 
    OneClassSVM().execute(idx=1, in_dir=create_synthetic_dataset, out_dir=out_dir)

    # Load results of the algorithm's execution
    output = pd.read_parquet(path.join(out_dir, '{}.gz'.format('01_OneClassSVM')))

    # Check output format (Check index, columns, monotonic index)
    assert output.shape == (500, 1)
    assert list(output.index.names) == OUTPUT_INDEX
    assert list(output.columns) == OUTPUT_COLUMNS
    assert output.index.is_monotonic
    np.testing.assert_array_equal(output.index.get_level_values(
        'sequence').unique(), [2])