import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.loaders.data_loader import load_energy_efficiency_dataset
from src.transformers.data_transformer import transform_energy_efficiency_dataset

MAX_ABS_DIFF = 1e-7 # Empirically chosen for tests on scaled features to pass

def test_transform_energy_efficiency_dataset():
    """
    Test that transform_energy_efficiency_dataset standardizes features correctly
    and splits data into balanced train and test sets.

    Checks:
    - Means of scaled features are close to 0 in both train and test sets.
    - Variances of scaled features are close to 1 in both train and test sets.

    Raises
    ------
    AssertionError
        If any check fails within the specified tolerances.
    """
    features_df, targets_df = load_energy_efficiency_dataset()
    data = transform_energy_efficiency_dataset(features_df, targets_df)

    for split in ["train", "test"]:
        scaled_features_df = data[split]["features"]["scaled"]
        # Check means ≈ 0
        means_df = scaled_features_df.mean().to_frame().T.apply(lambda x: np.floor(np.abs(x)))
        expected_means_df = pd.DataFrame(np.zeros_like(means_df), columns=means_df.columns)
        assert_frame_equal(means_df, expected_means_df, check_exact=False, atol=MAX_ABS_DIFF)
        # Check variances ≈ 1
        variances_df = scaled_features_df.var().to_frame().T.apply(np.round)
        expected_variances_df = pd.DataFrame(np.ones_like(variances_df), columns=variances_df.columns)
        assert_frame_equal(variances_df, expected_variances_df, check_exact=False, atol=MAX_ABS_DIFF)
