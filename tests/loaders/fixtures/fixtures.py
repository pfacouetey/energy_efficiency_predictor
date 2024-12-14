import pytest
import pandas as pd


@pytest.fixture()
def expected_features_means_df():
    return pd.DataFrame(
        {
            'Relative_Compactness': 0.76,
            'Surface_Area': 671.71,
            'Wall_Area': 318.50,
            'Roof_Area': 176.60,
            'Overall_Height': 5.25,
            'Orientation': 3.50,
            'Glazing_Area': 0.23,
            'Glazing_Area_Distribution': 2.81,
        },
        index=[0]
    )

@pytest.fixture()
def expected_targets_means_df():
    return pd.DataFrame(
        {
            'Heating_Load': 22.31,
            'Cooling_Load': 24.59,
        },
        index=[0]
    )

@pytest.fixture()
def expected_features_variances_df():
    return pd.DataFrame(
        {
            'Relative_Compactness': 0.01,
            'Surface_Area': 7759.16,
            'Wall_Area': 1903.27,
            'Roof_Area': 2039.96,
            'Overall_Height': 3.07,
            'Orientation': 1.25,
            'Glazing_Area': 0.02,
            'Glazing_Area_Distribution': 2.41,
        },
        index=[0]
    )

@pytest.fixture()
def expected_targets_variances_df():
    return pd.DataFrame(
        {
            'Heating_Load': 101.81,
            'Cooling_Load': 90.50,
        },
        index=[0]
    )
