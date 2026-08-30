"""
XGBoost regressor for AQI prediction.

Uses early stopping on a validation set to prevent overfitting,
which is especially important given the temporal autocorrelation
in hourly AQI data.
"""

from xgboost import XGBRegressor


DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

EARLY_STOPPING_ROUNDS = 20


def train(X_train, y_train, X_val=None, y_val=None, **overrides):
    """
    Train an XGBoost regressor with optional early stopping.

    Parameters
    ----------
    X_train, y_train : array-like
        Training features and target.
    X_val, y_val : array-like, optional
        Validation set for early stopping. If not provided,
        the model trains for the full ``n_estimators`` rounds.
    **overrides
        Any ``XGBRegressor`` kwargs to override defaults.

    Returns
    -------
    XGBRegressor
        The fitted model.
    """
    params = {**DEFAULT_PARAMS, **overrides}

    # Enable early stopping only when a validation set is available
    if X_val is not None and y_val is not None:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}
    else:
        fit_kwargs = {"verbose": False}

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, **fit_kwargs)
    return model
