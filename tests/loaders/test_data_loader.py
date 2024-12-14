from pandas.testing import assert_frame_equal

from src.loaders.data_loader import load_energy_efficiency_dataset
from tests.loaders.fixtures.fixtures import (expected_features_variances_df, expected_targets_variances_df,
                                             expected_targets_means_df, expected_features_means_df)

def test_load_energy_efficiency_dataset():
    """ Test the load_energy_efficiency_dataset function """
    features, targets = load_energy_efficiency_dataset()

    # Check feature columns
    expected_feature_columns = [
        'Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
        'Overall_Height', 'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution'
    ]
    assert list(features.columns) == expected_feature_columns, \
        f"Instead of {expected_feature_columns}, got {list(features.columns)}."

    # Check target columns
    expected_target_columns = ['Heating_Load', 'Cooling_Load']
    assert list(targets.columns) == expected_target_columns, \
        f"Instead of {expected_target_columns}, got {list(targets.columns)}."

    # Check data types
    assert all(features[col].dtype == float for col in features.columns)
    assert all(targets[col].dtype == float for col in targets.columns)

    # Check non-empty DataFrames
    assert not features.empty
    assert not targets.empty

    # Check non null values
    assert not features.isnull().values.any()
    assert not targets.isnull().values.any()

    print("All tests passed!")

def test_columns_statistics(
        expected_features_variances_df,
        expected_targets_variances_df,
        expected_features_means_df,
        expected_targets_means_df,
):
    """ Test the calculated mean and variance of the dataset. """
    features_df, targets_df = load_energy_efficiency_dataset()

    # Calculate actual means and variances
    actual_features_means_df = features_df.mean().to_frame().T
    actual_features_variances_df = features_df.var().to_frame().T
    actual_targets_means_df = targets_df.mean().to_frame().T
    actual_targets_variances_df = targets_df.var().to_frame().T

    # Compare DataFrames
    tol_float = 0.005  # Tolerance for floating point comparison
    assert_frame_equal(
        actual_features_means_df,
        expected_features_means_df,
        check_exact=False,
        atol=tol_float
    )
    assert_frame_equal(
        actual_features_variances_df,
        expected_features_variances_df,
        check_exact=False,
        atol=tol_float
    )
    assert_frame_equal(
        actual_targets_means_df,
        expected_targets_means_df,
        check_exact=False,
        atol=tol_float
    )
    assert_frame_equal(
        actual_targets_variances_df,
        expected_targets_variances_df,
        check_exact=False,
        atol=tol_float
    )

    print("All tests passed!")
