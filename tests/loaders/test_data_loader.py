from pandas.testing import assert_frame_equal

from src.loaders.data_loader import load_energy_efficiency_dataset
from tests.loaders.fixtures.fixtures import (
    expected_features_names,
    expected_targets_names,
    expected_features_variances_df,
    expected_targets_variances_df,
    expected_targets_means_df,
    expected_features_means_df,
)

MAX_ABS_DIFF = 5e-3  # Empirically chosen for test_columns_statistics to pass


def test_load_energy_efficiency_dataset(expected_features_names, expected_targets_names):
    """
    Test the loading of the energy efficiency dataset.

    This function validates the integrity, completeness, and correct structure of the loaded
    energy efficiency dataset. It checks that:
      1. Features and targets DataFrames are not empty.
      2. Features and targets do not contain any null values.
      3. Features DataFrame contains the expected feature columns in the correct order.
      4. Targets DataFrame contains the expected target columns in the correct order.
      5. All columns in both DataFrames have the correct data type (float).

    Raises
    ------
    AssertionError
        If any of the above checks fail.
    """
    features_df, targets_df = load_energy_efficiency_dataset()

    assert not features_df.empty, "Features DataFrame is empty."
    assert not targets_df.empty, "Targets DataFrame is empty."

    assert not features_df.isnull().values.any(), "Features DataFrame contains null values."
    assert not targets_df.isnull().values.any(), "Targets DataFrame contains null values."

    assert list(features_df.columns) == expected_features_names, (
        f"Expected feature columns {expected_features_names}, got {list(features_df.columns)}."
    )

    assert list(targets_df.columns) == expected_targets_names, (
        f"Expected target columns {expected_targets_names}, got {list(targets_df.columns)}."
    )

    assert all(features_df[col].dtype == float for col in features_df.columns), \
        "Not all feature columns are of type float."
    assert all(targets_df[col].dtype == float for col in targets_df.columns), \
        "Not all target columns are of type float."


def test_columns_statistics(
    expected_features_variances_df,
    expected_targets_variances_df,
    expected_features_means_df,
    expected_targets_means_df,
):
    """
    Compare computed statistical metrics of columns against expected values using mean and variance.
    Validates the integrity of data analysis by comparing actual and expected DataFrames.

    Parameters
    ----------
    expected_features_variances_df : pd.DataFrame
        DataFrame containing the expected variances for feature columns.
    expected_targets_variances_df : pd.DataFrame
        DataFrame containing the expected variances for target columns.
    expected_features_means_df : pd.DataFrame
        DataFrame containing the expected means for feature columns.
    expected_targets_means_df : pd.DataFrame
        DataFrame containing the expected means for target columns.

    Raises
    ------
    AssertionError
        If computed means or variances for features/targets differ from the expected values
        based on the specified tolerance.
    """
    features_df, targets_df = load_energy_efficiency_dataset()

    actual_features_means_df = features_df.mean().to_frame().T
    actual_targets_means_df = targets_df.mean().to_frame().T

    actual_features_variances_df = features_df.var().to_frame().T
    actual_targets_variances_df = targets_df.var().to_frame().T

    assert_frame_equal(
        actual_features_means_df,
        expected_features_means_df,
        check_exact=False,
        atol=MAX_ABS_DIFF,
    )
    assert_frame_equal(
        actual_features_variances_df,
        expected_features_variances_df,
        check_exact=False,
        atol=MAX_ABS_DIFF,
    )
    assert_frame_equal(
        actual_targets_means_df,
        expected_targets_means_df,
        check_exact=False,
        atol=MAX_ABS_DIFF,
    )
    assert_frame_equal(
        actual_targets_variances_df,
        expected_targets_variances_df,
        check_exact=False,
        atol=MAX_ABS_DIFF,
    )