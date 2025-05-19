import mlflow
import random
import logging
import numpy as np
import pandas as pd
from typing import Literal
from mlflow.models import infer_signature
from mlflow.exceptions import MlflowException
from sklearn.tree import DecisionTreeRegressor
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, KFold

SEED = 123
N_FOLDS = 5
DEFAULT_MAX_EVALS = 50
DECISIONTREE_HYPERPARAMETERS = {
    "criterion": hp.choice("criterion", ["squared_error", "friedman_mse", "absolute_error"]),
    "splitter": hp.choice("splitter", ["best", "random"]),
    "min_samples_split": hp.quniform("min_samples_split", 6, 20, 1),
    "max_depth": hp.quniform("max_depth", 3, 10, 1),
    "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 10, 1),
    "max_features": hp.uniform("max_features", 0.1, 1.0),
    "ccp_alpha": hp.uniform("ccp_alpha", 0.0, 0.5),
}

random.seed(SEED)


class CustomDecisionTreeRegressor(DecisionTreeRegressor):
    """
    Custom decision tree regressor that extends the functionality of the
    base DecisionTreeRegressor by storing the training data.

    This regressor behaves similarly to the standard DecisionTreeRegressor
    but retains the training data in object properties for any custom
    post-fit operations or analyses.

    Attributes
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature set used during the fit process.
    y_train : array-like of shape (n_samples,)
        Training target values corresponding to the training feature set.
    """
    def fit(self, X, y, **kwargs):
        self.X_train = X
        self.y_train = y
        return super().fit(X, y)

def custom_scorer(estimator, X, y):
    """
    Calculates performance metrics for a given estimator on both training and
    validation datasets. The function evaluates the estimator's performance
    by computing the minimum R^2 score and the maximum mean squared error
    across all output variables, for the validation and training datasets.

    Parameters
    ----------
    estimator : CustomDecisionTreeRegressor
        The fitted estimator whose performance metrics are to be calculated.
        Must be an instance of CustomDecisionTreeRegressor.
    X : array-like of shape (n_samples, n_features)
        The input features for the validation dataset.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target values for the validation dataset.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - "val_r2": Minimum R^2 score for the validation dataset.
        - "val_mse": Maximum mean squared error for the validation dataset.
        - "train_r2": Minimum R^2 score for the training dataset.
        - "train_mse": Maximum mean squared error for the training dataset.
    """
    if not isinstance(estimator, CustomDecisionTreeRegressor):
        raise ValueError("Estimator must be an instance of CustomDecisionTreeRegressor ...")

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
    Train a decision tree model using cross-validation and log results.

    This function trains a decision tree regressor using provided feature and target datasets
    along with specified hyperparameters. It leverages cross-validation to compute evaluation
    metrics and logs the model and associated data using MLflow. The function returns detailed
    training results, including performance metrics, the fitted model, feature importance, and
    MLflow signature.

    Parameters
    ----------
    features_df : pandas.DataFrame
        A DataFrame containing the input features for model training. Each row represents
        an instance, and each column represents a feature.

    targets_df : pandas.DataFrame
        A DataFrame containing the target variable(s) corresponding to the features provided.
        Each row corresponds to the target for the respective instance in the input data.

    hyperparameters : dict[str, float]
        Dictionary containing hyperparameters for the decision tree model. It must include:
        - "min_samples_split": Minimum number of samples required to split an internal node.
        - "min_samples_leaf": Minimum number of samples required to be at a leaf node.
        - "max_depth": Maximum depth of the tree.

    Returns
    -------
    dict[str, object] or None
        A dictionary is returned containing the following keys if the training succeeds:
        - "loss" : float
            Validation mean squared error score.
        - "val_mse_score" : float
            Mean squared error score on validation data from cross-validation.
        - "val_r2_score" : float
            R-squared (coefficient of determination) score for validation data.
        - "train_mse_score" : float
            Mean squared error score on training data from cross-validation.
        - "train_r2_score" : float
            R-squared (coefficient of determination) score for training data.
        - "status" : str
            Status indicating the training result, typically the value of `STATUS_OK`.
        - "model" : CustomDecisionTreeRegressor
            The trained decision tree model.
        - "features_importance" : pandas.DataFrame
            A DataFrame with feature names and their corresponding importance scores.
        - "signature" : mlflow.models.signature.ModelSignature
            Signature of the model created using the `infer_signature` function.

        Returns None if either `features_df` or `targets_df` is empty.
    """
    if features_df.empty or targets_df.empty:
        logging.error("At least one of the input dataframes is empty...")
        return None

    hyperparameters["min_samples_split"] = int(hyperparameters["min_samples_split"])
    hyperparameters["min_samples_leaf"] = int(hyperparameters["min_samples_leaf"])
    hyperparameters["max_depth"] = int(hyperparameters["max_depth"])

    model = CustomDecisionTreeRegressor(**hyperparameters, random_state=SEED)

    with mlflow.start_run(nested=True):

        logging.info("Started training of a decision tree...")
        cv = KFold(
            n_splits=N_FOLDS,
            random_state=SEED,
            shuffle=True,
        )
        scores = cross_validate(
            estimator=model,
            X=features_df,
            y=targets_df,
            scoring=custom_scorer,
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
        )
        model_fitted = model.fit(features_df, targets_df)
        results = {
            "model": model_fitted,
            "signature" : infer_signature(model_input=features_df, model_output=targets_df),
            "scores": scores,
            "val_r2_score": np.mean(scores["test_val_r2"]),
            "val_mse_score": np.mean(scores["test_val_mse"]),
            "train_r2_score": np.mean(scores["train_train_r2"]),
            "train_mse_score": np.mean(scores["train_train_mse"]),
            "features_importance": model_fitted.feature_importances_,
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
            "features_importance": pd.DataFrame(
                {
                    "feature": features_df.columns,
                    "importance": results["features_importance"],
                }
            ),
            "signature": results["signature"],
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

def decisiontree_regressor(
        features_df: pd.DataFrame = None,
        targets_df: pd.DataFrame = None,
        operation_mode: Literal["train", "test"] = "train",
        best_run: dict[str, object] = None,
        experiment_name: str = None,
) -> dict[str, object]|None:
    """
    Trains or tests a DecisionTree regressor model based on the provided operation mode.

    This function is designed to handle both training and testing workflows for a DecisionTree
    regressor. In training mode, the function performs hyperparameter tuning using Bayesian
    optimization and logs the best model to an experiment. In testing mode, predictions
    are generated using the provided model from the best run, and evaluation metrics are
    computed.

    Parameters
    ----------
    features_df : pd.DataFrame, optional
        Input features for training or testing the model. If empty, the function will log
        an error and return None.
    targets_df : pd.DataFrame, optional
        Target values corresponding to the features. If empty, the function will log an
        error and return None.
    operation_mode : {'train', 'test'}, default='train'
        Specifies the operation mode. 'train' for training the model, 'test' for evaluating
        an already trained model.
    best_run : dict[str, object], optional
        Dictionary containing information about the best run, including the trained model.
        This is used during testing mode. When in 'test' mode, the function will log an
        error and return None if `best_run` or the `model` within `best_run` is not
        provided.
    experiment_name : str, optional
        Name of the MLflow experiment under which the model and metrics will be logged
        during training. If not provided in 'train' mode, the function will log an error
        and return None.

    Returns
    -------
    dict[str, object] or None
        In 'train' mode, the function returns a dictionary containing the results of the
        best run, including hyperparameters, metrics, and the trained model. In 'test'
        mode, it returns a dictionary containing the evaluation metrics calculated on the
        test set. Returns None if an error occurs (e.g., missing input data, empty
        DataFrame, or invalid operation mode).
    """

    if features_df.empty or targets_df.empty:
        logging.error("At least one of the input dataFrames is empty ...")
        return None

    if operation_mode == "train":

        if experiment_name is None:
            logging.error("Experiment name is required for training mode...")
            return None

        logging.info("Defining objective function for hyperparameter tuning of DecisionTree...")

        def objective_function(hyperparameters: dict[str, float]) -> dict[str, float]:
            return train_model(
                features_df=features_df,
                targets_df=targets_df,
                hyperparameters=hyperparameters,
            )

        logging.info("Objective function defined successfully.")

        logging.info(f"Started hyperparameter tuning for DecisionTree using {experiment_name}...")
        logging.info("Number of evaluations: {}".format(DEFAULT_MAX_EVALS))
        set_or_create_experiment(experiment_name=experiment_name)
        with mlflow.start_run():
            trials = Trials()
            best_hyperparameters = fmin(
                fn=objective_function,
                space=DECISIONTREE_HYPERPARAMETERS,
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

        if best_run["model"] is None:
            logging.error("Model is not provided in the best run...")
            return None

        logging.info("Performing predictions on test set...")
        model = best_run["model"]
        pred_targets_array = model.predict(features_df)
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
