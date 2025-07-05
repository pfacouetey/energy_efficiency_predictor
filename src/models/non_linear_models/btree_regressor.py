import mlflow
import random
import logging
import numpy as np
import pandas as pd
from typing import Literal, Optional
from mlflow.models import infer_signature
from mlflow.exceptions import MlflowException
from sklearn.multioutput import MultiOutputRegressor
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, make_scorer

N_FOLDS = 5
SEED = 21062025
random.seed(SEED)
DEFAULT_MAX_EVALS = 50

# Hyperparameters' values were chosen empirically after doing runs, and understand how the model performs on the data
# By doing more experimentation, you could find better results
GB_HYPERPARAMETERS = {
    "n_estimators": hp.quniform("n_estimators", 50, 300, 10),
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),
    "max_depth": hp.choice("max_depth", [3, 5]),
    "max_features": hp.uniform("max_features", 0.1, 0.5),
    "subsample": hp.uniform("subsample", 0.1, 0.3)
}


def train_model(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    hyperparameters: dict,
) -> Optional[dict]:
    """
    Train a multi-output Gradient Boosting regressor with cross-validation and MLflow logging.

    Fits a MultiOutputRegressor wrapping a GradientBoostingRegressor on the provided features and targets,
    performs cross-validation with multiple scoring metrics (R², MSE, MAPE), and logs results to MLflow.
    Returns model, metrics, and feature importances.

    Parameters
    ----------
        features_df : pd.DataFrame
            DataFrame containing input features for training.
        targets_df : pd.DataFrame
            DataFrame containing target variables for training (multi-output supported).
        hyperparameters : dict
            Dictionary of hyperparameters for GradientBoostingRegressor.

    Returns
    -------
        dict or None
            Dictionary containing the trained model, cross-validation metrics, feature importances, and MLflow signature.
            Returns None if input dataframes are empty.

    Example
    -------
    >> results = train_model(X_train, Y_train, best_hyperparameters)

    Notes
    -----
        - Uses uniform averaging for multi-output metrics.
        - MAPE is reported as a percentage.
        - Features' importances are averaged across all targets.
        """
    if features_df.empty or targets_df.empty:
        logging.error("At least one of the input dataframes is empty...")
        return None

    for param in ["n_estimators", "min_samples_leaf"]:
        if param in hyperparameters:
            hyperparameters[param] = int(hyperparameters[param])
    if "max_depth" in hyperparameters and hyperparameters["max_depth"] is not None:
        hyperparameters["max_depth"] = int(hyperparameters["max_depth"])

    base_model = GradientBoostingRegressor(**hyperparameters, random_state=SEED)
    model = MultiOutputRegressor(base_model)

    # Define multi-metric scoring
    scoring = {
        'r2': make_scorer(r2_score, multioutput='uniform_average'),
        'mse': make_scorer(mean_squared_error, greater_is_better=False, multioutput='uniform_average'),
        'mape': make_scorer(mean_absolute_percentage_error, greater_is_better=False, multioutput='uniform_average'),
    }

    with mlflow.start_run(nested=True):
        logging.info("Started training of a multi-output gradient boosting regressor...")
        cv = KFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)

        scores = cross_validate(
            estimator=model,
            X=features_df,
            y=targets_df,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
        )
        model_fitted = model.fit(features_df, targets_df)

        # Aggregate feature importances (mean across targets)
        feature_importances = np.mean(
            [est.feature_importances_ for est in model_fitted.estimators_], axis=0
        )

        # All metrics: note that MSE and MAPE are negative (since greater_is_better=False)
        val_mse = -np.mean(scores["test_mse"])
        train_mse = -np.mean(scores["train_mse"])
        val_mape = -np.mean(scores["test_mape"])
        train_mape = -np.mean(scores["train_mape"])
        val_r2 = np.mean(scores["test_r2"])
        train_r2 = np.mean(scores["train_r2"])

        results = {
            "model": model_fitted,
            "signature": infer_signature(model_input=features_df, model_output=targets_df),
            "scores": scores,
            "val_r2_score": val_r2,
            "val_mse_score": val_mse,
            "val_mape_score": val_mape,
            "train_r2_score": train_r2,
            "train_mse_score": train_mse,
            "train_mape_score": train_mape,
            "features_importance": feature_importances,
        }
        logging.info("Training completed successfully.")

        logging.info("Start logging of model info...")
        mlflow.log_params(hyperparameters)
        mlflow.log_metric("train_mse_score", train_mse)
        mlflow.log_metric("train_r2_score", train_r2)
        mlflow.log_metric("train_mape_score", train_mape)
        mlflow.log_metric("val_mse_score", val_mse)
        mlflow.log_metric("val_r2_score", val_r2)
        mlflow.log_metric("val_mape_score", val_mape)
        mlflow.sklearn.log_model(results["model"], "model", signature=results["signature"])
        logging.info("Logging of model completed successfully.")

        return {
            # A way to guide to use the difference between train and validation as the model tuning target
            "loss": abs(train_mse - val_mse),
            "val_mse_score": val_mse,
            "val_r2_score": val_r2,
            "val_mape_score": val_mape,
            "train_mse_score": train_mse,
            "train_r2_score": train_r2,
            "train_mape_score": train_mape,
            "status": STATUS_OK,
            "model": results["model"],
            "features_importance": pd.DataFrame(
                {
                    "feature": features_df.columns,
                    "importance": feature_importances,
                }
            ),
            "signature": results["signature"],
        }

def set_or_create_experiment(experiment_name: str) -> None:
    """
    Set or create an MLflow experiment.

    Checks if an MLflow experiment with the given name exists. If not, creates it.
    If the experiment exists but is deleted, restores it. Sets the experiment as active.

    Parameters
    ----------
        experiment_name : str
            Name of the MLflow experiment to set or create.

    Returns
    -------
        None

    Example
    -------
    >> set_or_create_experiment("my_experiment")

    Notes
    -----
        - Requires MLflow tracking server to be running.
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

def optimized_btree_regressor(
    features_df: Optional[pd.DataFrame] = None,
    targets_df: Optional[pd.DataFrame] = None,
    operation_mode: Literal["train", "test"] = "train",
    best_run: Optional[dict] = None,
    experiment_name: Optional[str] = None,
) -> Optional[dict]:
    """
    Hyperparameter optimization and evaluation for multi-output Gradient Boosting regression.

    In 'train' mode, performs hyperparameter optimization using Hyperopt and cross-validation,
    logs the best model and metrics to MLflow, and returns the best run.
    In 'test' mode, evaluates the provided model on new test data and returns performance metrics.

    Parameters
    ----------
        features_df : pd.DataFrame, optional
            DataFrame containing input features.
        targets_df : pd.DataFrame, optional
            DataFrame containing target variables (multi-output supported).
        operation_mode: {'train', 'test'}, default='train'
            Whether to train (optimize and fit) or test (evaluate) the model.
        best_run : dict, optional
            Dictionary containing the best trained model and related info (required for 'test' mode).
        experiment_name : str, optional
            Name of the MLflow experiment (required for 'train' mode).

    Returns
    -------
        dict or None
            In 'train' mode: Dictionary with best model, metrics, and feature importances.
            In 'test' mode: Dictionary with test set metrics.
            Returns None if inputs are invalid.

    Example
    -------
    # Training
    >> best_run = optimized_btree_regressor(X_train, Y_train, operation_mode="train", experiment_name="exp1")
    # Testing
    >> test_results = optimized_btree_regressor(X_test, Y_test, operation_mode="test", best_run=best_run)

    Notes
    -----
        - Uses Hyperopt for hyperparameter search.
        - Supports multi-output regression via MultiOutputRegressor.
        - Logs all metrics and models to MLflow.
    """

    if features_df is None or targets_df is None or features_df.empty or targets_df.empty:
        logging.error("At least one of the input dataFrames is empty ...")
        return None

    if operation_mode == "train":
        if experiment_name is None:
            logging.error("Experiment name is required for training mode...")
            return None

        logging.info("Defining objective function for hyperparameter tuning of GradientBoosting...")

        def objective_function(hyperparameters: dict) -> dict:
            return train_model(
                features_df=features_df,
                targets_df=targets_df,
                hyperparameters=hyperparameters,
            )

        logging.info("Objective function defined successfully.")
        logging.info(f"Started hyperparameter tuning for GradientBoosting using {experiment_name}...")
        logging.info(f"Number of evaluations: {DEFAULT_MAX_EVALS}")
        set_or_create_experiment(experiment_name=experiment_name)
        with mlflow.start_run():
            trials = Trials()
            best_hyperparameters = fmin(
                fn=objective_function,
                space=GB_HYPERPARAMETERS,
                algo=tpe.suggest,
                max_evals=DEFAULT_MAX_EVALS,
                trials=trials,
                rstate=np.random.default_rng(SEED),
            )

            logging.info("Started logging of the best model...")
            best_run = sorted(trials.results, key=lambda x: x["loss"])[0]
            print(f"Training mse: {best_run['train_mse_score']}")
            print(f"Validation mse: {best_run['val_mse_score']}")
            print(20 * "-")
            print(f"Training mape: {best_run['train_mape_score']}")
            print(f"Validation mape: {best_run['val_mape_score']}")
            print(20 * "-")
            print(f"Training r2: {best_run['train_r2_score']}")
            print(f"Validation r2: {best_run['val_r2_score']}")
            mlflow.log_params(best_hyperparameters)
            mlflow.log_metric("best_val_mse", best_run["val_mse_score"])
            mlflow.log_metric("best_val_mape", best_run["val_mape_score"])
            mlflow.log_metric("best_val_r2", best_run["val_r2_score"])
            mlflow.sklearn.log_model(best_run["model"], "model", signature=best_run["signature"])
            logging.info("Best model logged successfully.")

        logging.info("Experiment completed successfully.")
        return best_run

    if operation_mode == "test":
        if not best_run or best_run.get("model") is None:
            logging.error("Model is not provided in the best run...")
            return None

        logging.info("Performing predictions on test set...")
        model = best_run["model"]
        pred_targets_array = model.predict(features_df)

        # Use uniform_average for multi-target
        mse_score = mean_squared_error(targets_df, pred_targets_array, multioutput="uniform_average")
        mape_score = mean_absolute_percentage_error(targets_df, pred_targets_array, multioutput="uniform_average") * 100
        r2_score_val = r2_score(targets_df, pred_targets_array, multioutput="uniform_average")

        results = {
            "mse_score": mse_score,
            "mape_score": mape_score,
            "r2_score": r2_score_val,
        }
        logging.info("Prediction completed successfully.")
        return results

    return None
