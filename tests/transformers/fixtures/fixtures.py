import pytest
import pandas as pd


@pytest.fixture()
def expected_features_means_df():
    return pd.DataFrame(
        {
            "Relative_Compactness": 0.0,
            "Surface_Area": 0.0,
            "Wall_Area": 0.0,
            "Roof_Area": 0.0,
            "Overall_Height": 0.0,
            "Orientation": 0.0,
            "Glazing_Area": 0.0,
            "Glazing_Area_Distribution": 0.0,
        },
        index=[0]
    )

@pytest.fixture()
def expected_features_variances_df():
    return pd.DataFrame(
        {
            "Relative_Compactness": 1.0,
            "Surface_Area": 1.0,
            "Wall_Area": 1.0,
            "Roof_Area": 1.0,
            "Overall_Height": 1.0,
            "Orientation": 1.0,
            "Glazing_Area": 1.0,
            "Glazing_Area_Distribution": 1.0,
        },
        index=[0]
    )

@pytest.fixture()
def expected_noisy_df():
    return pd.DataFrame(
        {
            "Feature_1": [0.457184, 2.141489, 2.710699],
            "Feature_2": [4.498672, 4.246852, 6.825718],
        }
    )