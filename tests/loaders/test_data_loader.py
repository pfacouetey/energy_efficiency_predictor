import pandas as pd

from src.loaders.data_loader import load_energy_efficiency_dataset
from tests.loaders.fixtures.fixtures import df_expected_features_variances, df_expected_targets_variances, df_expected_targets_means, df_expected_features_means  # noqa: F401

def test_load_energy_efficiency_dataset():
    """ Test the load_energy_efficiency_dataset function """
    features, targets = load_energy_efficiency_dataset()

    # Test feature columns
    expected_feature_columns = [
        'Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
        'Overall_Height', 'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution'
    ]
    assert list(features.columns) == expected_feature_columns

    # Test target columns
    expected_target_columns = ['Heating_Load', 'Cooling_Load']
    assert list(targets.columns) == expected_target_columns

    # Test data types
    assert all(features[col].dtype == float for col in features.columns)
    assert all(targets[col].dtype == float for col in targets.columns)

    # Test non-empty DataFrames
    assert not features.empty
    assert not targets.empty

def test_columns_statistics(
        df_expected_features_variances,
        df_expected_features_means,
        df_expected_targets_variances,
        df_expected_targets_means,
):
    """ Test the calculated mean and variance of the dataset. """
    features, targets = load_energy_efficiency_dataset()

    # Calculate actual means and variances
    df_actual_features_means = features.mean().to_frame().T
    df_actual_features_variances = features.var().to_frame().T
    df_actual_targets_means = targets.mean().to_frame().T
    df_actual_targets_variances = targets.var().to_frame().T

    # Compare DataFrames
    pd.testing.assert_frame_equal(df_actual_features_means, df_expected_features_means, check_exact=False, atol=0.1)
    pd.testing.assert_frame_equal(df_actual_features_variances, df_expected_features_variances, check_exact=False, atol=0.1)
    pd.testing.assert_frame_equal(df_actual_targets_means, df_expected_targets_means, check_exact=False, atol=0.1)
    pd.testing.assert_frame_equal(df_actual_targets_variances, df_expected_targets_variances, check_exact=False, atol=0.1)
    print("All tests passed!")
