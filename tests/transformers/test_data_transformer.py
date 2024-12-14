import numpy as np
from pandas.testing import assert_frame_equal

from src.loaders.data_loader import load_energy_efficiency_dataset
from src.transformers.data_transformer import transform_energy_efficiency_dataset
from tests.transformers.fixtures.fixtures import expected_features_means_df, expected_features_variances_df

def test_transform_energy_efficiency_dataset(
        expected_features_means_df,
        expected_features_variances_df,
):
    """ Test the transform_energy_efficiency_dataset function """
    features_df, targets_df = load_energy_efficiency_dataset()
    data_dict = transform_energy_efficiency_dataset(
        features_df=features_df,
        targets_df=targets_df
    )
    scaled_train_features_df = data_dict["train"]["features"]
    scaled_test_features_df = data_dict["test"]["features"]
    scaled_val_features_df = data_dict["val"]["features"]

    # Calculate actual means for train, test, and validation sets
    actual_scaled_train_features_means_df = scaled_train_features_df.mean().to_frame().T.apply(lambda x: np.floor(np.abs(x)))
    actual_scaled_test_features_means_df = scaled_test_features_df.mean().to_frame().T.apply(lambda x: np.floor(np.abs(x)))
    actual_scaled_val_features_means_df = scaled_val_features_df.mean().to_frame().T.apply(lambda x: np.floor(np.abs(x)))

    # Calculate actual variances for train, test, and validation sets
    actual_scaled_train_features_variances_df = scaled_train_features_df.var().to_frame().T.apply(np.round)
    actual_scaled_test_features_variances_df = scaled_test_features_df.var().to_frame().T.apply(np.round)
    actual_scaled_val_features_variances_df = scaled_val_features_df.var().to_frame().T.apply(np.round)

    # Compare DataFrames
    tol_float = 1e-06  # Tolerance for floating point comparison
    assert_frame_equal(
        actual_scaled_train_features_means_df,
        expected_features_means_df,
        check_exact=False,
        atol=tol_float
    )
    assert_frame_equal(
        actual_scaled_test_features_means_df,
        expected_features_means_df,
        check_exact=False,
        atol=tol_float
    )
    assert_frame_equal(
        actual_scaled_val_features_means_df,
        expected_features_means_df,
        check_exact=False,
        atol=tol_float
    )
    assert_frame_equal(
        actual_scaled_train_features_variances_df,
        expected_features_variances_df,
        check_exact=False,
        atol=tol_float
    )
    assert_frame_equal(
        actual_scaled_test_features_variances_df,
        expected_features_variances_df,
        check_exact=False,
        atol=tol_float
    )
    assert_frame_equal(
        actual_scaled_val_features_variances_df,
        expected_features_variances_df,
        check_exact=False,
        atol=tol_float
    )
    print("All tests passed!")
