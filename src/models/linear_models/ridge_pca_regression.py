import logging
import pandas as pd
from typing import Literal
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def ridge_pca_regression(
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        operation_mode: Literal["train", "validate"] = "train",
        n_components: int = None,
        prediction_models: dict[str, object] = None,
) -> dict[str, object] | None:
    pass