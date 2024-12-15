import logging
import pandas as pd
from typing import Literal
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def pca_regression(
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        operation_mode: Literal["train", "validate"] = "train",
        n_components: int = None,
        prediction_models: dict[str, object] = None,
) -> dict[str, object] | None:
    """
    Perform PCA on feature dataset and train Linear Regression models for each target variable.

    This function takes a dataset of features and a dataset of targets, performs Principal Component
    Analysis (PCA) to reduce the dimensionality of the feature set, and trains linear regression models
    for each target variable. If the feature dataset is empty, no processing occurs, and the function
    returns None.

    Parameters:
    features_df : pd.DataFrame
        A pandas DataFrame containing the features to be used in the PCA and regression analysis.
    targets_df : pd.DataFrame
        A pandas DataFrame containing the target variables for which regression models will be trained.

    Returns:
    dict[str, object] | None
        A dictionary mapping target variable names to their corresponding trained Linear Regression
        model objects. Returns None if the features dataset is empty.
    """

    if not features_df.empty :

        if operation_mode == "train" and not targets_df.empty:
            logging.info("Performing PCA on dataset...")
            pca_features_df = PCA(n_components=0.9).fit_transform(features_df)
            logging.info("PCA completed successfully.")

            logging.info("Training Linear Regression model...")
            models_dict = {"n_components": pca_features_df.shape[1]}
            for target_name in targets_df.columns:
                model = LinearRegression().fit(pca_features_df, targets_df[target_name])
                models_dict[target_name] = {}
                models_dict[target_name]["model"] = model
                models_dict[target_name]["r_squared"] = model.score(pca_features_df, targets_df[target_name])
                models_dict[target_name]["mse"] = mean_squared_error(
                    model.predict(pca_features_df), targets_df[target_name])
            logging.info("Training complete successfully.")

            return models_dict

        elif (
                operation_mode == "validate" and n_components is not None and prediction_models is not None and (
                not targets_df.empty)
        ):
            logging.info("Performing PCA on dataset...")
            pca_features_df = PCA(n_components=n_components).fit_transform(features_df)
            logging.info("PCA completed successfully.")

            logging.info("Started doing prediction...")
            models_dict = {"n_components": n_components}
            for target_name in [key for key in prediction_models.keys() if key != "n_components"]:
                model = prediction_models[target_name]["model"]
                models_dict[target_name] = {}
                models_dict[target_name]["model"] = model
                models_dict[target_name]["r_squared"] = model.score(pca_features_df, targets_df[target_name])
                models_dict[target_name]["mse"] = mean_squared_error(
                    model.predict(pca_features_df), targets_df[target_name])
            logging.info("Prediction successful.")

            return models_dict

    else:
        logging.error("Failed to train or perform prediction...")
        return None
