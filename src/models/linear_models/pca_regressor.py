import random
import logging
import itertools
import numpy as np
import pandas as pd
from typing import Literal
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

N_FOLDS = 5
SEED = 21062025
random.seed(SEED)
N_COMPONENTS = 0.9  # Fraction of variance explained by principal components
N_COMPONENTS_FIXED = 5  # For optimized PCA: value known from previous computation done on PCA


class CustomStatsmodelsOLS:
    """
    OLS Regression model that stores training data for use in custom scoring.

    This class wraps statsmodels' OLS regression and adds attributes to keep the
    training features and targets, enabling custom scoring functions that require
    access to the training data.

    Example
    -------
    >> model = CustomStatsmodelsOLS()
    """

    def __init__(self):
        """
        Initialize the CustomStatsmodelsOLS instance.
        """
        super().__init__()
        self.model = None
        self.X_train = None
        self.y_train = None
        self.feature_names = None  # Track columns used in training

    def fit(self, X, y):
        """
        Fit the OLS model and store the training data.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Training data.
        y : array-like or pd.Series
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        X_const = sm.add_constant(X, has_constant='skip')
        self.X_train = X_const
        self.y_train = y
        self.feature_names = X_const.columns  # Save for use in predict
        self.model = sm.OLS(y, X_const).fit()
        return self

    def predict(self, X):
        """
        Predict target values using the trained OLS model.

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Data for which to predict target values.

        Returns
        -------
        y_pred : np.ndarray or pd.Series
            Predicted target values.
        """
        X_const = sm.add_constant(X, has_constant='skip')
        # Ensure columns are in the same order as during training
        X_const = X_const.reindex(columns=self.feature_names, fill_value=0)
        return self.model.predict(X_const)

    @property
    def bic(self):
        """Direct access to statsmodels' BIC calculation"""
        return self.model.bic

def custom_scorer(estimator, X, y):
    """
    Custom scoring function for regression models.

    Calculates Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE)
    for both validation and training sets.

    Parameters
    ----------
    estimator : CustomLinearRegression
        The regression model being evaluated.
    X: array-like or pd.DataFrame
        Validation features.
    y : array-like or pd.Series
        Validation targets.

    Returns
    -------
    dict
        Dictionary containing validation and training MSE and MAPE scores.

    Example
    -------
    >> scores = custom_scorer(model, X_val, y_val)
    """
    if not isinstance(estimator, CustomStatsmodelsOLS):
        raise ValueError("Estimator must be an instance of CustomStatsmodelsOLS.")
    y_val_pred = estimator.predict(X)
    val_mse = mean_squared_error(y_true=y, y_pred=y_val_pred)
    val_mape = mean_absolute_percentage_error(y_true=y, y_pred=y_val_pred) * 100  # as percentage
    y_train_pred = estimator.predict(estimator.X_train)
    train_mse = mean_squared_error(y_true=estimator.y_train, y_pred=y_train_pred)
    train_mape = mean_absolute_percentage_error(y_true=estimator.y_train, y_pred=y_train_pred) * 100  # as percentage
    return {
        "val_mse": val_mse,
        "val_mape": val_mape,
        "train_mse": train_mse,
        "train_mape": train_mape,
    }

def pca_regressor(
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        operation_mode: Literal["train", "test"] = "train",
        train_results: dict[str, object] = None,
) -> dict[str, object] | None:
    """
    Perform PCA-based linear regression modeling and evaluation, using statsmodels OLS.

    In 'train' mode, fits PCA on features, trains a linear regression model for each target,
    and evaluates with cross-validation using custom scoring (R^2, MSE, MAPE).
    In 'test' mode, uses the trained models and PCA object to predict and evaluate on new data.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing feature columns.
    targets_df : pd.DataFrame
        DataFrame containing target columns.
    operation_mode : {'train', 'test'}, default='train'
        Mode of operation. In 'train', fits models; in 'test', evaluates on new data.
    train_results : dict, optional
        Dictionary containing trained models and PCA object (required for 'test' mode).

    Returns
    -------
    dict or None
        In 'train' mode: Dictionary with trained models, PCA object, and cross-validation metrics.
        In 'test' mode: Dictionary with test set evaluation metrics for each target, and predictions.
        Returns None if input DataFrames are empty or arguments are invalid.

    Example
    -------
    # Training
    >> train_results = pca_regressor(features_df, targets_df, operation_mode="train")
    # Testing
    >> test_results = pca_regressor(
    test_features_df, test_targets_df, operation_mode="test", train_results=train_results)

    Notes
    -----
    - MAPE is reported as a percentage.
    - Requires statsmodels, scikit-learn and pandas.
    """
    if features_df.empty or targets_df.empty:
        logging.error("At least one of the input dataFrames is empty.")
        return None

    if operation_mode == "train":
        cv = KFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)
        train_results = {}
        for target_name in targets_df.columns:
            val_mses, val_mapes = [], []
            train_mses, train_mapes = [], []
            for train_idx, val_idx in cv.split(features_df):
                X_train, X_val = features_df.iloc[train_idx], features_df.iloc[val_idx]
                y_train, y_val = targets_df[target_name].iloc[train_idx], targets_df[target_name].iloc[val_idx]

                # Perform PCA
                pca = PCA(n_components=N_COMPONENTS)
                X_train_pca = pca.fit_transform(X_train)
                X_val_pca = pca.transform(X_val)
                X_train_pca = pd.DataFrame(X_train_pca, columns=[f"PC{i + 1}" for i in range(X_train_pca.shape[1])])
                X_val_pca = pd.DataFrame(X_val_pca, columns=[f"PC{i + 1}" for i in range(X_val_pca.shape[1])])

                # Reset indices to ensure alignment for statsmodels
                X_train_pca, X_val_pca = X_train_pca.reset_index(drop=True), X_val_pca.reset_index(drop=True)
                y_train, y_val = y_train.reset_index(drop=True), y_val.reset_index(drop=True)

                # Fit model and score
                model = CustomStatsmodelsOLS().fit(X_train_pca, y_train)
                scores = custom_scorer(model, X_val_pca, y_val)
                val_mses.append(scores["val_mse"])
                val_mapes.append(scores["val_mape"])
                train_mses.append(scores["train_mse"])
                train_mapes.append(scores["train_mape"])

            # Fit PCA and model on all data for deployment
            pca_full = PCA(n_components=N_COMPONENTS)
            X_full_pca = pca_full.fit_transform(features_df.reset_index(drop=True))
            X_full_pca = pd.DataFrame(X_full_pca, columns=[f"PC{i + 1}" for i in range(X_full_pca.shape[1])])
            y_full = targets_df[target_name].reset_index(drop=True)
            model_full = CustomStatsmodelsOLS().fit(X_full_pca, y_full)
            train_results[target_name] = {
                "model": model_full,
                "val_mse_score": np.mean(val_mses),
                "val_mape_score": np.mean(val_mapes),
                "train_mse_score": np.mean(train_mses),
                "train_mape_score": np.mean(train_mapes),
                "predictions": pd.DataFrame(model_full.predict(X_full_pca), columns=[target_name]),
                "pca_object": pca_full,  # Save per-target PCA for test phase
            }
        return train_results

    if operation_mode == "test" and train_results is not None:
        test_results = {}
        for target_name in targets_df.columns:
            pca = train_results[target_name]["pca_object"]
            X_test_pca = pca.transform(features_df)
            X_test_pca = pd.DataFrame(X_test_pca, columns=[f"PC{i + 1}" for i in range(X_test_pca.shape[1])])
            model = train_results[target_name]["model"]
            y_true = targets_df[target_name]
            y_pred = model.predict(X_test_pca)
            test_results[target_name] = {
                "mse_score": mean_squared_error(y_true, y_pred),
                "mape_score": mean_absolute_percentage_error(y_true, y_pred) * 100,
                "predictions": pd.DataFrame(y_pred, columns=[target_name])
            }
        return test_results

    return None

def optimized_pca_regressor(
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        operation_mode: Literal["train", "test"] = "train",
        train_results: dict[str, object] = None,
) -> dict[str, object] | None:
    """
        Perform PCA-based linear regression with BIC-optimized feature selection.

        In 'train' mode:
            1. Fits PCA on full training data
            2. Performs exhaustive BIC-based selection of principal components (1-5 PCs)
            3. Uses cross-validation to compute training/validation errors
            4. Stores best model with selected PCs

        In 'test' mode: Uses trained models for prediction with selected PCs

        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame containing feature columns.
        targets_df : pd.DataFrame
            DataFrame containing target columns.
        operation_mode : {'train', 'test'}, default='train'
            Mode of operation. In 'train', fits models; in 'test', evaluates on new data.
        train_results : dict, optional
            Dictionary containing trained models, PCA objects, and selected PCs
            (required for 'test' mode).

        Returns
        -------
        dict or None
            In 'train' mode: Dictionary with keys for each target containing:
                - model: Trained OLS model
                - pca_object: Fitted PCA transformer
                - selected_pcs: Indices of significant principal components
                - validation metrics (MSE/MAPE)
                - predictions: Training predictions
            In 'test' mode: Dictionary with test metrics and predictions per target
            Returns None if inputs are empty or invalid

        Example
        -------
        # Training
        >> train_results = optimized_pca_regressor(
            features_train, targets_train, operation_mode="train")

        # Testing
        >> test_results = optimized_pca_regressor(
            features_test, targets_test,
            operation_mode="test",
            train_results=train_results)

        Notes
        -----
        - Uses exhaustive search over PC subsets (1-5 components) based on BIC criteria
        - MAPE is reported as percentage
        - Requires statsmodels, scikit-learn, and pandas
        - During testing, only selected PCs are transformed and used for prediction
    """
    if features_df.empty or targets_df.empty:
        logging.error("Input DataFrames cannot be empty")
        return None

    if operation_mode == "train":
        train_results = {}
        cv = KFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)

        for target_name in targets_df.columns:
            # Step 1: Fit PCA on FULL training data
            pca = PCA(n_components=N_COMPONENTS_FIXED)
            X_train_pca = pca.fit_transform(features_df)
            X_train_pca = pd.DataFrame(X_train_pca,
                                       columns=[f"PC{i + 1}" for i in range(X_train_pca.shape[1])])
            y_train = targets_df[target_name].reset_index(drop=True)

            # Step 2: BIC-based feature selection on FULL data
            best_bic = np.inf
            best_subset = None
            best_model = None

            n_pcs = X_train_pca.shape[1]
            all_subsets = list(itertools.chain.from_iterable(
                itertools.combinations(range(n_pcs), r)
                for r in range(1, min(6, n_pcs + 1))
            ))

            for subset in all_subsets:
                X_subset = X_train_pca.iloc[:, list(subset)]
                X_subset = X_subset.reset_index(drop=True)
                model = CustomStatsmodelsOLS().fit(X_subset, y_train)

                if model.bic < best_bic:
                    best_bic = model.bic
                    best_subset = list(subset)
                    best_model = model

            # Step 3: Cross-validation for error estimation
            val_mses, val_mapes = [], []
            train_mses, train_mapes = [], []

            for train_idx, val_idx in cv.split(features_df):
                # Prepare fold data
                X_train_fold, X_val_fold = features_df.iloc[train_idx], features_df.iloc[val_idx]
                y_train_fold = targets_df[target_name].iloc[train_idx].reset_index(drop=True)
                y_val_fold = targets_df[target_name].iloc[val_idx].reset_index(drop=True)

                # Apply SAME PCA transformation
                X_train_fold_pca = pca.transform(X_train_fold)
                X_val_fold_pca = pca.transform(X_val_fold)
                X_train_fold_pca = pd.DataFrame(
                    X_train_fold_pca,
                    columns=[f"PC{i + 1}" for i in range(X_train_fold_pca.shape[1])])
                X_val_fold_pca = pd.DataFrame(
                    X_val_fold_pca,
                    columns=[f"PC{i + 1}" for i in range(X_val_fold_pca.shape[1])])

                # Select SAME PC subset
                X_train_fold_subset = X_train_fold_pca.iloc[:, best_subset].reset_index(drop=True)
                X_val_fold_subset = X_val_fold_pca.iloc[:, best_subset].reset_index(drop=True)

                # Train model on fold training data
                fold_model = CustomStatsmodelsOLS().fit(X_train_fold_subset, y_train_fold)

                # Compute errors
                train_pred = fold_model.predict(X_train_fold_subset)
                val_pred = fold_model.predict(X_val_fold_subset)

                train_mses.append(mean_squared_error(y_train_fold, train_pred))
                train_mapes.append(mean_absolute_percentage_error(y_train_fold, train_pred) * 100)
                val_mses.append(mean_squared_error(y_val_fold, val_pred))
                val_mapes.append(mean_absolute_percentage_error(y_val_fold, val_pred) * 100)

            # Store results
            train_results[target_name] = {
                "model": best_model,
                "pca_object": pca,
                "selected_pcs": best_subset,
                "train_mse_score": np.mean(train_mses),
                "train_mape_score": np.mean(train_mapes),
                "val_mse_score": np.mean(val_mses),
                "val_mape_score": np.mean(val_mapes),
                "predictions": pd.DataFrame(best_model.predict(X_train_pca.iloc[:, best_subset]),
                                            columns=[target_name])
            }

        return train_results

    elif operation_mode == "test" and train_results is not None:
        test_results = {}
        for target_name, model_data in train_results.items():
            pca = model_data["pca_object"]
            selected_pcs = model_data["selected_pcs"]

            # Transform and select PCs
            X_test_pca = pca.transform(features_df)
            X_test_pca = pd.DataFrame(X_test_pca,
                                      columns=[f"PC{i + 1}" for i in range(X_test_pca.shape[1])])
            X_test_subset = X_test_pca.iloc[:, selected_pcs]

            # Predict and evaluate
            model = model_data["model"]
            y_true = targets_df[target_name]
            y_pred = model.predict(X_test_subset)

            test_results[target_name] = {
                "mse_score": mean_squared_error(y_true, y_pred),
                "mape_score": mean_absolute_percentage_error(y_true, y_pred) * 100,
                "predictions": pd.DataFrame(y_pred, columns=[target_name])
            }

        return test_results

    return None