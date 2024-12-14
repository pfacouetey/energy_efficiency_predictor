import logging
import pandas as pd

from ucimlrepo import fetch_ucirepo

def load_energy_efficiency_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the Energy Efficiency dataset from the UCI Machine Learning Repository, separate
    its covariates and targets, and return them as two distinct dataframes. The dataset
    consists of features related to various architectural properties alongside heating
    and cooling load targets. Column names for both features and targets are reformatted
    for better clarity. Logs status of dataset loading and processing operations.

    :return: A tuple containing two pandas DataFrames: the cleaned features dataframe and
        the cleaned targets dataframe. If the dataset is unavailable or empty, returns
        two empty dataframes.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    logging.info("Loading Energy Efficiency dataset...")

    # Fetch the dataset from the UCI Machine Learning Repository
    energy_efficiency_df = fetch_ucirepo(id=242)

    # Separate features and targets when fetching is successful and the dataset is not empty.
    if not energy_efficiency_df.empty:
        logging.info("Separating covariates from target...")
        features_df = energy_efficiency_df.data.features
        features_df = features_df.rename(
            columns={
                'X1': 'Relative_Compactness',
                'X2': 'Surface_Area',
                'X3': 'Wall_Area',
                'X4': 'Roof_Area',
                'X5': 'Overall_Height',
                'X6': 'Orientation',
                'X7': 'Glazing_Area',
                'X8': 'Glazing_Area_Distribution'
            },
        ).map(float)
        targets_df = energy_efficiency_df.data.targets
        targets_df = targets_df.rename(
            columns={
                'Y1': 'Heating_Load',
                'Y2': 'Cooling_Load',
            },
        ).map(float)
        logging.info("Energy Efficiency dataset loaded successfully.")

        return features_df, targets_df

    else:
        logging.error("Failed to load Energy Efficiency dataset...")
        return pd.DataFrame(), pd.DataFrame()
