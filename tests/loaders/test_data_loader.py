from pandas.testing import assert_frame_equal

from src.loaders.data_loader import load_energy_efficiency_dataset
from tests.loaders.fixtures.fixtures import (expected_features_variances_df, expected_targets_variances_df,
                                             expected_targets_means_df, expected_features_means_df)

TOL_FLOAT = 0.005


def test_load_energy_efficiency_dataset():
    """
    Test the loading of the energy efficiency dataset.

    This function validates the integrity, completeness, and correct
    structure of the loaded energy efficiency dataset. It performs
    checks on the following aspects:

    1. Ensures that features and targets dataframes are not empty.
    2. Verifies that features and targets do not contain any null
       values.
    3. Validates that features dataframe contains the expected feature
       columns with the correct order.
    4. Confirms that targets dataframe contains the expected target
       columns with the correct order.
    5. Ensures that all columns in both features and targets dataframes
       have the correct data type (float).

    Raises
    ------
    AssertionError
        If any of the above checks fail, an assertion error is raised
        indicating the specific mismatch or issue.
    """
    features, targets = load_energy_efficiency_dataset()

    assert not features.empty
    assert not targets.empty

    assert not features.isnull().values.any()
    assert not targets.isnull().values.any()

    expected_feature_columns = [
        'Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
        'Overall_Height', 'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution'
    ]
    assert list(features.columns) == expected_feature_columns, \
        f"Instead of {expected_feature_columns}, got {list(features.columns)}."

    expected_target_columns = ['Heating_Load', 'Cooling_Load']
    assert list(targets.columns) == expected_target_columns, \
        f"Instead of {expected_target_columns}, got {list(targets.columns)}."

    assert all(features[col].dtype == float for col in features.columns)
    assert all(targets[col].dtype == float for col in targets.columns)

    print("All tests passed!")

def test_columns_statistics(
        expected_features_variances_df,
        expected_targets_variances_df,
        expected_features_means_df,
        expected_targets_means_df,
):
    """
        Compares computed statistical metrics of columns against expected values
        using mean and variance calculations. This function validates the integrity
        of data analysis on the energy efficiency dataset by comparing actual and
        expected dataframes through a series of assertions.

        Parameters:
        ----------
        expected_features_variances_df : pd.DataFrame
            The dataframe containing the expected variances for feature columns.

        expected_targets_variances_df : pd.DataFrame
            The dataframe containing the expected variances for target columns.

        expected_features_means_df : pd.DataFrame
            The dataframe containing the expected means for feature columns.

        expected_targets_means_df : pd.DataFrame
            The dataframe containing the expected means for target columns.

        Raises:
        ------
        AssertionError
            If computed means or variances for features/targets differ from the
            expected values based on the specified tolerance.
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
        atol=TOL_FLOAT
    )
    assert_frame_equal(
        actual_features_variances_df,
        expected_features_variances_df,
        check_exact=False,
        atol=TOL_FLOAT
    )
    assert_frame_equal(
        actual_targets_means_df,
        expected_targets_means_df,
        check_exact=False,
        atol=TOL_FLOAT
    )
    assert_frame_equal(
        actual_targets_variances_df,
        expected_targets_variances_df,
        check_exact=False,
        atol=TOL_FLOAT
    )
    print("All tests passed!")
