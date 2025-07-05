import random
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SEED = 21062025
TEST_SIZE = 0.2
random.seed(SEED)


def transform_energy_efficiency_dataset(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
) -> dict[str, dict[str, pd.DataFrame]] | None:
    """
    Splits and standardizes the Energy Efficiency dataset for model training and evaluation.

    This function takes feature and target DataFrames, performs a stratified train-test split,
    and standardizes features columns using scikit-learn's StandardScaler.
    The result is a structured dictionary
    containing both the original and standardized feature sets for training and testing, as well as
    the corresponding targets.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing the feature columns:
        ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
         'Overall_Height', 'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution'].
    targets_df : pd.DataFrame
        DataFrame containing the target columns:
        ['Heating_Load', 'Cooling_Load'].

    Returns
    -------
    dict[str, dict[str, pd.DataFrame]] or None
        A nested dictionary with the following structure:
            {
                "train": {
                    "features": {
                        "original": <pd.DataFrame>,
                        "scaled": <pd.DataFrame>
                    },
                    "targets": <pd.DataFrame>
                },
                "test": {
                    "features": {
                        "original": <pd.DataFrame>,
                        "scaled": <pd.DataFrame>
                    },
                    "targets": <pd.DataFrame>
                }
            }
        Returns None if either input DataFrame is empty.

    Example
    -------
    >> data = transform_energy_efficiency_dataset(features_df, targets_df)

    Notes
    -----
    - The function returns None if the input DataFrames are empty.
    """
    if features_df.empty or targets_df.empty:
        logging.error("Input DataFrames cannot be empty")
        return None

    logging.info("Splitting data into training and test sets...")

    train_features_df, test_features_df, train_targets_df, test_targets_df = train_test_split(
        features_df, targets_df,
        test_size=TEST_SIZE,
        random_state=SEED,
    )

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_features_df)
    scaled_test = scaler.transform(test_features_df)

    return {
        "train": {
            "features": {
                "original": train_features_df,
                "scaled": pd.DataFrame(
                    scaled_train,
                    columns=features_df.columns,
                    index=train_features_df.index
                )
            },
            "targets": train_targets_df
        },
        "test": {
            "features": {
                "original": test_features_df,
                "scaled": pd.DataFrame(
                    scaled_test,
                    columns=features_df.columns,
                    index=test_features_df.index
                )
            },
            "targets": test_targets_df
        }
    }
