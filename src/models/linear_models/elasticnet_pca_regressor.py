import mlflow
import random
import logging
import numpy as np
import pandas as pd
from typing import Literal
from mlflow import MlflowException
from sklearn.decomposition import PCA
from mlflow.models import infer_signature
from sklearn.linear_model import ElasticNet
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, KFold

SEED = 123
N_FOLDS = 5
DEFAULT_MAX_EVALS = 30
N_COMPONENTS = 0.9
ELASTICNET_HYPERPARAMETERS = {
    "alpha": hp.loguniform("alpha", np.log(0.01), np.log(100.0)),
    "l1_ratio": hp.uniform("l1_ratio", 0.001, 1.0),
}

np.random.seed(SEED)
random.seed(SEED)

class CustomElasticNet(ElasticNet):
    """
    A custom implementation of the ElasticNet regression model.

    This subclass of ElasticNet is designed to enhance the
    base functionality by retaining the training data for
    further analysis or processing. It overrides the `fit`
    method from the ElasticNet class to store the input
    features and target values as attributes of the model.

    Attributes
    ----------
    X_train : array-like of shape (n_samples, n_features)
        The training input samples. This attribute stores the
        feature matrix passed to the `fit` method.
    y_train : array-like of shape (n_samples,)
        The target values. This attribute stores the target
        vector passed to the `fit` method.
    """
    def fit(self, X, y, **kwargs):
        self.X_train = X
        self.y_train = y
        return super().fit(X, y)

def custom_scorer(estimator, X, y):
    """
    Evaluates the performance of a given estimator on both validation and training data
    using custom scoring metrics. The function calculates the minimum coefficient of
    determination (R^2) and the maximum mean squared error (MSE) for multi-output
    predictions. It ensures the custom estimator meets the required type.

    Parameters
    ----------
    estimator : CustomElasticNet
        The estimator whose performance is to be evaluated. Must be an instance of
        the CustomElasticNet class and contain `X_train` and `y_train` attributes for
        training data.
    X : array-like of shape (n_samples, n_features)
        Validation feature dataset.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True target values for the validation dataset.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - "val_r2": Minimum R^2 score from validation predictions.
        - "val_mse": Maximum MSE from validation predictions.
        - "train_r2": Minimum R^2 score from training predictions.
        - "train_mse": Maximum MSE from training predictions.

    Raises
    ------
    ValueError
        If the provided estimator is not an instance of the CustomElasticNet class.
    """

    if not isinstance(estimator, CustomElasticNet):
        raise ValueError("Estimator must be an instance of CustomElasticNet ...")

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
    Train a linear regression model with PCA feature transformation and evaluate
    its performance using cross-validation. The method utilizes Ridge and Lasso
    regularization via a custom ElasticNet model, internally logging metrics and
    the trained model using MLflow.

    Parameters
    ----------
    features_df : pd.DataFrame
        The input DataFrame containing the features for training. Each row
        represents a sample, and each column represents a feature.

    targets_df : pd.DataFrame
        The target DataFrame containing the outputs corresponding to the input
        features. It should have the same number of rows as `features_df`.

    hyperparameters : dict[str, float]
        The dictionary of hyperparameter values to configure the custom ElasticNet
        model. Keys represent the names of hyperparameters and values represent
        their corresponding numeric values.

    Returns
    -------
    dict[str, object] or None
        A dictionary containing model results and evaluation metrics if training
        and validation are successful. The keys include:

        - "loss": Mean squared error score on the validation set.
        - "val_mse_score": Averaged validation mean squared error.
        - "val_r2_score": Averaged validation R-squared score.
        - "train_mse_score": Averaged training mean squared error.
        - "train_r2_score": Averaged training R-squared score.
        - "status": Status indicating the completion of the process.
        - "model": Trained instance of the custom ElasticNet model.
        - "signature": Inferred MLflow model signature.
        - "n_components": Number of PCA components used in feature transformation.

        If either `features_df` or `targets_df` is empty during input validation,
        the function logs an error and returns None.
    """

    if features_df.empty or targets_df.empty:
        logging.error("At least one of the input dataframes is empty...")
        return None

    model = CustomElasticNet(**hyperparameters, random_state=SEED)

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
            n_splits=N_FOLDS,
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

def set_or_create_experiment(
        experiment_name: str,
) -> None:
    """
    Set or create an experiment in MLflow.

    This function checks if an experiment with the given name exists in MLflow.
    If the experiment exists and is marked as deleted, it restores the experiment.
    If the experiment is active, it sets the experiment as the current one.
    Otherwise, it creates a new experiment with the given name. If any error
    occurs during the process, it raises an MlflowException.

    Parameters
    ----------
    experiment_name : str
        The name of the experiment to set, restore, or create.

    Returns
    -------
    None
        This function does not return any value.

    Raises
    ------
    MlflowException
        If an error occurs while interacting with the MLflow tracking API.
    """
    client = mlflow.tracking.MlflowClient()
    try:
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment and experiment.lifecycle_stage == "deleted":
            print(f"Experiment {experiment_name} is deleted. Restoring it...")
            client.restore_experiment(experiment.experiment_id)
            print(f"Experiment {experiment_name} successfully restored.")

        elif experiment:
            print(f"Experiment {experiment_name} exists and is active.")

        else:
            print(f"Creating new experiment: {experiment_name}")
            mlflow.create_experiment(experiment_name)

        mlflow.set_experiment(experiment_name)

    except MlflowException as e:
        print(f"An error occurred while handling the experiment: {e}")
        raise

def elasticnet_pca_regressor(
        features_df: pd.DataFrame = None,
        targets_df: pd.DataFrame = None,
        operation_mode: Literal["train", "test"] = "train",
        best_run: dict[str, object] = None,
        experiment_name: str = None,
) -> dict[str, object]|None:
    """
    ElasticNet PCA Regressor.

    This function implements a pipeline for training and testing a regression model
    that uses ElasticNet regularization combined with Principal Component Analysis (PCA).
    The function supports two operation modes: "train" for hyperparameter tuning
    and model training, and "test" for leveraging the best trained model to make
    predictions on a test dataset. During the training, hyperparameters are
    optimized using a Bayesian optimization approach.

    Parameters
    ----------
    features_df : pd.DataFrame, optional
        DataFrame containing the feature set to be utilized for training or testing.
        Each row corresponds to an observation, while columns represent individual features.

    targets_df : pd.DataFrame, optional
        DataFrame containing the target values. Each row corresponds to the
        target value of the respective observation in `features_df`.

    operation_mode : Literal["train", "test"], default="train"
        Specifies the operation mode of the function:
        - "train": Conducts hyperparameter tuning, trains the ElasticNet model,
          and logs the best model to a tracking server.
        - "test": Uses the best trained model and parameters to make predictions
          on the test dataset.

    best_run : dict[str, object], optional
        A dictionary containing the best results obtained from training,
        including the trained model and PCA components, required for the "test" mode.

    experiment_name : str, optional
        The name of the experiment, crucial in "train" mode for logging and tracking
        the hyperparameter optimization process and training progress.

    Returns
    -------
    dict[str, object] or None
        - In "train" mode: Returns a dictionary containing the best hyperparameters,
          trained model, performance metrics, and metadata for the ElasticNet model.
        - In "test" mode: Returns a dictionary with evaluation metrics (`mse_score`,
          `r2_score`) calculated on the test dataset.
        - Returns None if invalid inputs or conditions are encountered.
    """

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
        set_or_create_experiment(experiment_name=experiment_name)
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

    if operation_mode == "test":

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

        if best_run["model"] is None:
            logging.error("Model is not provided in the best run...")
            return None

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
