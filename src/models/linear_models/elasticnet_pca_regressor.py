import mlflow
import random
import logging
import numpy as np
import pandas as pd
from typing import Literal
from sklearn.decomposition import PCA
from mlflow.models import infer_signature
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, KFold

SEED = 123
DEFAULT_MAX_EVALS = 30
N_COMPONENTS = 0.9
ELASTICNET_HYPERPARAMETERS = {
    "alpha": hp.loguniform("alpha", np.log(0.01), np.log(100.0)),
    "l1_ratio": hp.uniform("l1_ratio", 0.001, 1.0),
}

np.random.seed(SEED)
random.seed(SEED)

class CustomMultiOutputRegressor(MultiOutputRegressor):
    """
    Extends the functionality of MultiOutputRegressor by maintaining a reference
    to the training data. Specifically, stores the feature matrix and target values
    used during training for later access. This can be useful for debugging or
    other custom tasks that require access to the training data after fitting.

    Attributes
    ----------
    X_train : array-like of shape (n_samples, n_features)
        The feature matrix used during the training process.
    y_train : array-like of shape (n_samples, n_outputs)
        The target values used during the training process.
    """
    def fit(self, X, y, **kwargs):
        self.X_train = X
        self.y_train = y
        return super().fit(X, y)

def custom_scorer(estimator, X, y):
    """
    Evaluates performance metrics for a given estimator using both training and validation data.

    The function computes the minimum R-squared value and the maximum mean squared error
    of predictions for the validation and training datasets. It requires the estimator to
    be an instance of `CustomMultiOutputRegressor`.

    Parameters
    ----------
    estimator : CustomMultiOutputRegressor
        A trained estimator of type `CustomMultiOutputRegressor`. The estimator must have
        been fit with training data, and the attributes `X_train` and `y_train` should be
        available.
    X : array-like
        Feature matrix for the validation dataset, used to generate predictions.
    y : array-like
        Ground truth target values for the validation dataset.

    Returns
    -------
    dict
        A dictionary containing the following evaluation metrics:
        - "val_r2": Minimum R-squared value for validation predictions.
        - "val_mse": Maximum mean squared error for validation predictions.
        - "train_r2": Minimum R-squared value for training predictions.
        - "train_mse": Maximum mean squared error for training predictions.

    Raises
    ------
    ValueError
        If the provided `estimator` is not an instance of `CustomMultiOutputRegressor`.
    """
    if not isinstance(estimator, CustomMultiOutputRegressor):
        raise ValueError("Estimator must be an instance of CustomMultiOutputRegressor ...")

    y_val_pred = estimator.predict(X)
    val_r2 = np.min(
        r2_score(y_true=y, y_pred=y_val_pred, multioutput="raw_values")
    )
    val_mse = np.max(
        mean_squared_error(y_true=y, y_pred=y_val_pred, multioutput="raw_values")
    )

    y_train_pred = estimator.predict(estimator.X_train)
    train_r2 = np.min(
        r2_score(y_true=estimator.y_train, y_pred=y_train_pred, multioutput="raw_values")
    )
    train_mse = np.max(
        mean_squared_error(y_true=estimator.y_train, y_pred=y_train_pred, multioutput="raw_values")
    )

    return {
        "val_r2": val_r2,
        "val_mse": val_mse,
        "train_r2": train_r2,
        "train_mse": train_mse,
    }

def train_model(
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        hyperparameters: dict[str, float],
) -> dict[str, object] | None:
    """
    Train a linear regression model with Ridge and Lasso regularization using PCA-transformed
    features and cross-validation scoring. Log the trained model, parameters, and metrics
    to MLflow for experiment tracking. Handles cases with empty input DataFrames.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing the feature columns used for training. Each column corresponds
        to a feature, and each row corresponds to an observation.
    targets_df : pd.DataFrame
        DataFrame containing the target columns to predict. Each column corresponds to a target
        variable, and each row corresponds to an observation that aligns with `features_df`.
    hyperparameters : dict[str, float]
        Dictionary containing hyperparameters for the ElasticNet model. Accepted keys and
        values align with parameters of `ElasticNet` from scikit-learn (e.g., alpha, l1_ratio).

    Returns
    -------
    dict[str, object] | None
        A dictionary containing the trained model, performance metrics, PCA components count,
        and related information if training succeeds. Returns None if either input DataFrame
        is empty. The dictionary includes the following keys:
        - "loss" : float
          Validation mean squared error (val_mse_score).
        - "val_mse_score" : float
          Mean validation mean squared error from cross-validation.
        - "val_r2_score" : float
          Mean validation R^2 score from cross-validation.
        - "train_mse_score" : float
          Mean training mean squared error from cross-validation.
        - "train_r2_score" : float
          Mean training R^2 score from cross-validation.
        - "status" : str
          Status of the training process, typically "STATUS_OK".
        - "model" : CustomMultiOutputRegressor
          The fitted model after training.
        - "signature" : mlflow.models.signature.ModelSignature
          Input-output signature of the logged MLFlow model.
        - "n_components" : int
          Number of principal components retained by PCA.
    """
    if features_df.empty or targets_df.empty:
        logging.error("At least one of the input dataframes is empty...")
        return None

    model = CustomMultiOutputRegressor(estimator=ElasticNet(**hyperparameters, random_state=SEED))

    with mlflow.start_run(nested=True):

        logging.info("Performing PCA on dataset...")
        pca_features_array = PCA(n_components=N_COMPONENTS).fit_transform(features_df)
        pca_features_df = pd.DataFrame(
            data=pca_features_array,
            columns=[f"PC{i + 1}" for i in range(pca_features_array.shape[1])],
        )
        logging.info("PCA completed successfully.")

        logging.info("Started training of a linear regression model with Ridge and Lasso regularization...")
        cv = KFold(
            n_splits=5,
            random_state=SEED,
            shuffle=True,
        )
        scores = cross_validate(
            estimator=model,
            X=pca_features_df,
            y=targets_df,
            scoring=custom_scorer,
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
        )
        results = {
            "model": model.fit(pca_features_df, targets_df),
            "signature" : infer_signature(model_input=pca_features_df, model_output=targets_df),
            "scores": scores,
            "val_r2_score": np.mean(scores["test_val_r2"]),
            "val_mse_score": np.mean(scores["test_val_mse"]),
            "train_r2_score": np.mean(scores["train_train_r2"]),
            "train_mse_score": np.mean(scores["train_train_mse"]),
        }
        logging.info("Training completed successfully.")

        logging.info("Start logging of model info...")
        mlflow.log_params(hyperparameters)
        mlflow.log_metric("train_mse", results["train_mse_score"])
        mlflow.log_metric("train_r2_score", results["train_r2_score"])
        mlflow.log_metric("val_mse_score", results["val_mse_score"])
        mlflow.log_metric("val_r2_score", results["val_r2_score"])
        mlflow.sklearn.log_model(results["model"], "model", signature=results["signature"])
        logging.info("Logging of model completed successfully.")

        return {
            "loss": results["val_mse_score"],
            "val_mse_score": results["val_mse_score"],
            "val_r2_score": results["val_r2_score"],
            "train_mse_score": results["train_mse_score"],
            "train_r2_score": results["train_r2_score"],
            "status": STATUS_OK,
            "model": results["model"],
            "signature": results["signature"],
            "n_components": int(pca_features_df.shape[1]),
        }

def elasticnet_pca_regressor(
        features_df: pd.DataFrame = None,
        targets_df: pd.DataFrame = None,
        operation_mode: Literal["train", "test"] = "train",
        best_run: dict[str, object] = None,
        experiment_name: str = None,
) -> dict[str, object]|None:
    """"""

    if features_df.empty or targets_df.empty:
        logging.error("At least one of the input dataFrames is empty ...")
        return None

    if operation_mode == "train":

        if experiment_name is None:
            logging.error("Experiment name is required for training mode...")
            return None

        logging.info("Defining objective function for hyperparameter tuning of ElasticNet...")
        def objective_function(hyperparameters: dict[str, float]) -> dict[str, float]:
            return train_model(
                features_df=features_df,
                targets_df=targets_df,
                hyperparameters=hyperparameters,
            )
        logging.info("Objective function defined successfully.")

        logging.info(f"Started hyperparameter tuning for ElasticNet using {experiment_name}...")
        logging.info("Number of evaluations: {}".format(DEFAULT_MAX_EVALS))
        mlflow.set_experiment(experiment_name=experiment_name)
        with mlflow.start_run():
            trials = Trials()
            best_hyperparameters = fmin(
                fn=objective_function,
                space=ELASTICNET_HYPERPARAMETERS,
                algo=tpe.suggest,
                max_evals=DEFAULT_MAX_EVALS,
                trials=trials,
                rstate=np.random.default_rng(SEED),
            )

            logging.info("Started logging of the best model...")
            best_run = sorted(trials.results, key=lambda x: x["loss"])[0]
            print(f"Training mse: {best_run['train_mse_score']}")
            print(f"Training r2: {best_run['train_r2_score']}")
            print(f"Validation mse / loss: {best_run['val_mse_score']}")
            print(f"Validation r2: {best_run['val_r2_score']}")
            mlflow.log_params(best_hyperparameters)
            mlflow.log_metric("best_val_mse", best_run["val_mse_score"])
            mlflow.log_metric("best_val_r2", best_run["val_r2_score"])
            mlflow.sklearn.log_model(best_run["model"], "model", signature=best_run["signature"])
            logging.info("Best model logged successfully.")

        logging.info("Experiment completed successfully.")
        return best_run

    elif operation_mode == "test":

        if best_run["n_components"] is None:
            logging.error("PCA components count is not provided in the best run...")
            return None

        logging.info("Performing PCA on dataset...")
        pca_features_array = PCA(n_components=best_run["n_components"]).fit_transform(features_df)
        pca_features_df = pd.DataFrame(
            data=pca_features_array,
            columns=[f"PC{i + 1}" for i in range(pca_features_array.shape[1])],
        )
        logging.info("PCA completed successfully.")

        logging.info("Performing predictions on test set...")
        model = best_run["model"]
        pred_targets_array = model.predict(pca_features_df)
        results = {
            "mse_score": np.max(
                mean_squared_error(y_true=targets_df, y_pred=pred_targets_array, multioutput="raw_values")
            ),
            "r2_score": np.min(
                r2_score(y_true=targets_df, y_pred=pred_targets_array, multioutput="raw_values")
            ),
        }
        logging.info("Prediction completed successfully.")
        return results
