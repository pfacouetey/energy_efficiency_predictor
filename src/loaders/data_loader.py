import logging
import pandas as pd

from ucimlrepo import fetch_ucirepo

def load_energy_efficiency_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the Energy Efficiency dataset.

    This function fetches the Energy Efficiency dataset using the `fetch_ucirepo`
    utility, processes it to separate features (covariates) from targets, and then
    renames columns for better readability. The dataset contains information about
    various building properties and their corresponding energy efficiency measures.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - The first DataFrame contains the features (covariates) with columns:
          'Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
          'Overall_Height', 'Orientation', 'Glazing_Area',
          'Glazing_Area_Distribution'.
        - The second DataFrame contains the targets with columns: 'Heating_Load',
          'Cooling_Load'.
    """
    logging.info("Loading Energy Efficiency dataset...")

    energy_efficiency_df = fetch_ucirepo(id=242)

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

    logging.error("Failed to load Energy Efficiency dataset...")
    return pd.DataFrame(), pd.DataFrame()
