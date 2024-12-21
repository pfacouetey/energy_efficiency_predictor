import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def transform_energy_efficiency_dataset(
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
) -> dict[str, dict[str, pd.DataFrame]] | None:
    """
    Transforms and partitions a dataset for energy efficiency analysis by splitting it into
    training, testing, and validation subsets and applying standardization to feature
    values. The function expects non-empty pandas DataFrames for both features and
    targets.

    :param features_df: Features dataset to be split and transformed.
    :type features_df: pd.DataFrame
    :param targets_df: Targets dataset corresponding to the features, to be split.
    :type targets_df: pd.DataFrame
    :return: A dictionary containing standardized training, testing, and validation
        datasets for features and corresponding targets if input dataFrames are non-empty.
        Otherwise, returns None.
    :rtype: dict[str, dict[str, pd.DataFrame]] | None
    """
    if not features_df.empty and not targets_df.empty:
        features_columns = features_df.columns.tolist()
        targets_columns = targets_df.columns.tolist()

        logging.info("Splitting data into train, test and validation sets...")
        train_features_df, test_val_features_df, train_targets_df, test_val_targets_df = train_test_split(
            features_df,
            targets_df,
            test_size=0.35,
            random_state=123,
        )
        test_features_df, val_features_df, test_targets_df, val_targets_df = train_test_split(
            test_val_features_df,
            test_val_targets_df,
            test_size=0.5,
            random_state=123,
        )
        logging.info("Data splitting completed successfully.")

        logging.info("Standardizing features...")
        scaler = StandardScaler()
        scaled_train_features_df = scaler.fit_transform(train_features_df)
        scaled_test_features_df = scaler.transform(test_features_df)
        scaled_val_features_df = scaler.transform(val_features_df)
        logging.info("Standardization completed successfully.")

        data_dict = {
            'train': {
                'features': pd.DataFrame(scaled_train_features_df, columns=features_columns),
                'targets': pd.DataFrame(train_targets_df, columns=targets_columns)
            },
            'test': {
                'features': pd.DataFrame(scaled_test_features_df, columns=features_columns),
                'targets': pd.DataFrame(test_targets_df, columns=targets_columns)
            },
            'val': {
                'features': pd.DataFrame(scaled_val_features_df, columns=features_columns),
                'targets': pd.DataFrame(val_targets_df, columns=targets_columns)
            }
        }
        return data_dict
    logging.error("At least one of the input dataFrames is empty ...")
    return None
