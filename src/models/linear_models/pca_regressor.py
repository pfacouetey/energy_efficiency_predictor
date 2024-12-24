import random
import logging
import numpy as np
import pandas as pd
from typing import Literal
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, KFold

N_COMPONENTS = 0.9
SEED = 123

np.random.seed(SEED)
random.seed(SEED)


class CustomLinearRegression(LinearRegression):
    """
    Extends the core LinearRegression model to integrate enhanced functionality.

    This class adds the capability to preserve the training data as attributes,
    which can be useful in scenarios where training features or labels need
    to be accessed after the model fitting step. Apart from this, it inherits
    all the functionalities of the conventional LinearRegression model and is
    compatible with its established methods for training and prediction.

    Attributes
    ----------
    X_train : ndarray or DataFrame
        Training features used during the fit method.
    y_train : ndarray or Series
        Training labels used during the fit method.
    """
    def fit(self, X, y, **kwargs):
        self.X_train = X
        self.y_train = y
        return super().fit(X, y)

def custom_scorer(estimator, X, y):
    """
    Evaluate a custom linear regression model using validation and training data metrics.

    This function assesses the performance of a `CustomLinearRegression` estimator
    by computing the R-squared (R2) and Mean Squared Error (MSE) metrics for both
    validation and training datasets. The function ensures that the `estimator`
    provided is an instance of the `CustomLinearRegression` class before proceeding.

    Parameters
    ----------
    estimator : CustomLinearRegression
        The trained instance of `CustomLinearRegression` used to compute predictions
        and evaluate model performance.
    X : array-like of shape (n_samples, n_features)
        Validation feature set used to generate predictions for evaluation.
    y : array-like of shape (n_samples,)
        True target values corresponding to the validation feature set.

    Returns
    -------
    dict
        A dictionary containing the computed performance metrics:
        - "val_r2": R-squared score for the validation dataset.
        - "val_mse": Mean squared error for the validation dataset.
        - "train_r2": R-squared score for the training dataset.
        - "train_mse": Mean squared error for the training dataset.

    Raises
    ------
    ValueError
        If the `estimator` is not an instance of `CustomLinearRegression`.
    """
    if not isinstance(estimator, CustomLinearRegression):
        raise ValueError("Estimator must be an instance of CustomLinearRegression ...")

    y_val_pred = estimator.predict(X)
    val_r2 = r2_score(y_true=y, y_pred=y_val_pred)
    val_mse = mean_squared_error(y_true=y, y_pred=y_val_pred)

    y_train_pred = estimator.predict(estimator.X_train)
    train_r2 = r2_score(y_true=estimator.y_train, y_pred=y_train_pred)
    train_mse = mean_squared_error(y_true=estimator.y_train, y_pred=y_train_pred)

    return {
        "val_r2": val_r2,
        "val_mse": val_mse,
        "train_r2": train_r2,
        "train_mse": train_mse,
    }

def pca_regressor(
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        operation_mode: Literal["train", "test"] = "train",
        prediction_models: dict[str, object] = None,
) -> dict[str, object] | None:
    """
    Performs Principal Component Analysis (PCA) on the input features and applies linear
    regression or predictions based on the specified operation mode. In "train" mode,
    this function trains linear regression models on the PCA-transformed features, performs
    cross-validation, and computes evaluation metrics. In "test" mode, it applies PCA
    using the specified number of components and evaluates the provided prediction
    models.

    Parameters
    ----------
    features_df : pd.DataFrame
        A DataFrame containing the feature variables for PCA and regression.

    targets_df : pd.DataFrame
        A DataFrame containing the target variables for regression or prediction
        evaluation.

    operation_mode : Literal["train", "test"], optional
        Specifies whether the function should train models ("train") or evaluate
        predictions using pre-trained models ("test"). Defaults to "train".

    prediction_models : dict[str, object], optional
        A dictionary containing pre-trained models and PCA component information.
        Required when operation_mode is "test". The dictionary should include:
        - Trained models for each target variable with the key as the target
          variable name and value as a dictionary containing the following:
            - "model": A trained regression model.
            - Any other necessary components such as metrics or settings used
              during training execution.
        - "n_components": The number of PCA components utilized for preprocessing.
          This must match the number of PCA components used during training.

    Returns
    -------
    dict[str, object] or None
        - In "train" mode:
          Returns a dictionary where each key corresponds to a target variable and
          its value contains:
            - "model": Trained linear regression model fitted on PCA-transformed
              features.
            - "val_r2_score": Mean cross-validation R^2 score on validation sets.
            - "val_mse_score": Mean cross-validation Mean Squared Error (MSE) on
              validation sets.
            - "train_r2_score": Mean R^2 score on training sets during
              cross-validation.
            - "train_mse_score": Mean MSE on training sets during cross-validation.
          Additionally, the dictionary includes a key "n_components", indicating
          the number of components used in PCA during training.

        - In "test" mode:
          Returns a dictionary where each key represents a target variable and
          its value contains:
            - "r2_score": The R^2 score achieved on test data.
            - "mse_score": The Mean Squared Error (MSE) achieved on test data.

        - If any error conditions are met (e.g., empty input data frames, invalid
          operation mode, missing components or models), the function logs the
          error and returns None.
    """
    if features_df.empty or targets_df.empty:
        logging.error("At least one of the input dataFrames is empty ...")
        return None

    if operation_mode == "train":

        logging.info("Performing PCA on dataset...")
        pca_features_array = PCA(n_components=N_COMPONENTS).fit_transform(features_df)
        pca_features_df = pd.DataFrame(
            data=pca_features_array,
            columns=[f"PC{i + 1}" for i in range(pca_features_array.shape[1])],
        )
        logging.info("PCA completed successfully.")

        logging.info("Creating cross-validation sets...")
        cv = KFold(
            n_splits=5,
            random_state=SEED,
            shuffle=True,
        )
        logging.info("Cross-validation sets created successfully.")

        logging.info("Training a linear regression model on PCA components...")
        results = {}
        for target_name in targets_df.columns:
            model = CustomLinearRegression()
            scores = cross_validate(
                estimator=model,
                X=pca_features_df,
                y=targets_df[target_name],
                scoring=custom_scorer,
                cv=cv,
                n_jobs=-1,
                return_train_score=True,
            )
            results[target_name] = {
                "model": model.fit(X=pca_features_df, y=targets_df[target_name], ),
                "val_r2_score": np.mean(scores["test_val_r2"]),
                "val_mse_score": np.mean(scores["test_val_mse"]),
                "train_r2_score": np.mean(scores["train_train_r2"]),
                "train_mse_score": np.mean(scores["train_train_mse"]),
            }
        logging.info("Training completed successfully.")

        results["n_components"] = pca_features_df.shape[1]
        return results

    elif operation_mode == "test" and prediction_models is not None:

        if prediction_models["n_components"] is None:
            logging.error("PCA components count is not provided in the prediction models...")
            return None

        logging.info("Performing PCA on dataset...")
        pca_features_array = PCA(n_components=prediction_models["n_components"]).fit_transform(features_df)
        pca_features_df = pd.DataFrame(
            data=pca_features_array,
            columns=[f"PC{i + 1}" for i in range(pca_features_array.shape[1])],
        )
        logging.info("PCA completed successfully.")

        logging.info("Performing predictions on test set...")
        results = {}
        for target_name in targets_df.columns:
            model = prediction_models[target_name]["model"]
            results[target_name] = {
                "r2_score": model.score(X=pca_features_df, y=targets_df[target_name]),
                "mse_score": mean_squared_error(
                    y_pred=model.predict(pca_features_df),
                    y_true=targets_df[target_name],
                ),
            }
        logging.info("Prediction completed successfully.")
        return results
