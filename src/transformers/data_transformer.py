import random
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SEED = 123
TEST_SIZE = 0.3

np.random.seed(SEED)
random.seed(SEED)


def add_gaussian_noise(
        features_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Adds Gaussian noise to the input DataFrame.

    This function takes a DataFrame of features and applies Gaussian noise to each
    value. It generates noise using a normal distribution with a mean of 0.0 and a
    standard deviation of 0.5. If the input DataFrame is empty, it logs a warning
    and returns None. The resulting DataFrame, with added Gaussian noise, is
    returned.

    Args:
        features_df (pd.DataFrame): The input DataFrame containing feature values
        whose entries will have Gaussian noise applied.

    Returns:
        pd.DataFrame | None: A DataFrame with Gaussian noise added to its values,
        or None if the input DataFrame is empty.

    Raises:
        This function does not explicitly raise any errors, but unexpected issues
        could arise due to DataFrame processing or noise generation.
    """
    if features_df.empty:
        logging.warning("No features to add Gaussian noise to.")
        return None

    logging.info("Generating Gaussian noise...")
    noise_df = pd.DataFrame(
        data=np.random.normal(
            loc=0.0,
            scale=0.5,
            size=(
                features_df.shape[0],
                features_df.shape[1],
            )
        ),
        columns=features_df.columns.tolist(),
        index=features_df.index,
    )
    logging.info("Gaussian noise generation completed successfully.")
    return features_df + noise_df

def transform_energy_efficiency_dataset(
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
) -> dict[str, dict[str, pd.DataFrame]] | None:
    """
    Transforms the energy efficiency dataset by splitting into training and test sets,
    adding Gaussian noise to the training set, and standardizing the features.

    This function takes in features and targets DataFrames, splits them into
    training and test sets, applies Gaussian noise to the training features,
    and standardizes both the original and noisy datasets. The standardized
    datasets are returned as a structured dictionary for further processing.

    Parameters:
        features_df (pd.DataFrame): DataFrame containing the features of the dataset.
        targets_df (pd.DataFrame): DataFrame containing the target values of the dataset.

    Returns:
        dict[str, dict[str, pd.DataFrame]] | None: A dictionary containing the structured
        data for original and noisy datasets, each with standardized training and testing
        features and targets. Returns None if any input DataFrame is empty.

    Raises:
        ValueError: If either `features_df` or `targets_df` is empty.
    """
    if features_df.empty or targets_df.empty:
        logging.error("At least one of the input dataFrames is empty ...")
        return None

    logging.info("Splitting data into training, and test sets...")
    train_features_df, test_features_df, train_targets_df, test_targets_df = train_test_split(
            features_df,
            targets_df,
            test_size=TEST_SIZE,
            random_state=123,
        )
    logging.info("Data splitting completed successfully.")

    logging.info("Adding Gaussian noise to training set...")
    noisy_train_features_df = add_gaussian_noise(features_df=train_features_df)
    logging.info("Gaussian noise addition completed successfully.")

    logging.info("Standardizing features...")
    scaler1, scaler2 = StandardScaler(), StandardScaler()
    data = {
        "train": {
            "features": {
                "original": pd.DataFrame(
                    data=scaler1.fit_transform(train_features_df),
                    columns=train_features_df.columns,
                ),
                "noise": pd.DataFrame(
                    data=scaler2.fit_transform(noisy_train_features_df),
                    columns=noisy_train_features_df.columns,
                ),
            },
            "targets": train_targets_df,
        },
        "test": {
            "features": {
                "original": pd.DataFrame(
                    data=scaler1.transform(test_features_df),
                    columns=test_features_df.columns,
                ),
                "noise": pd.DataFrame(
                    data=scaler2.transform(test_features_df),
                    columns=test_features_df.columns,
                ),
            },
            "targets": test_targets_df,
        }
    }
    logging.info("Standardization completed successfully.")
    return data
