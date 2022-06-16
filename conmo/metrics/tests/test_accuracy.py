from os import path

import numpy as np
import pandas as pd

from conmo.metrics import Accuracy

OUTPUT_INDEX = ['fold']
OUTPUT_COLUMNS = ['01_KerasAutoencoder']

def test_accuracy(tmp_path_factory, create_synthetic_ground_truth, create_synthetic_results):
    # Generate temporary output directory for feeding pipeline
    out_dir = tmp_path_factory.mktemp('output')

    # Instantiate and calculate accuracy
    Accuracy().calculate(1, OUTPUT_COLUMNS,
                         create_synthetic_ground_truth, create_synthetic_results, out_dir)

    # Load results of the metric's calculation
    output = pd.read_parquet(path.join(out_dir, '{}.gz'.format('01_Accuracy')))

    # Check output format (Check index, columns, monotonic index)
    assert output.shape == (2, 1)
    assert list(output.index.names) == OUTPUT_INDEX
    assert list(output.columns) == OUTPUT_COLUMNS
    np.testing.assert_array_equal(output.index.get_level_values(
        'fold').unique(), [1, 2])

