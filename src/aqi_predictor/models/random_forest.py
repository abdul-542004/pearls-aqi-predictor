"""
Random Forest regressor for AQI prediction.

Wraps scikit-learn's RandomForestRegressor with project-standard
`train()` interface so the training pipeline can call all models
the same way.
"""

from sklearn.ensemble import RandomForestRegressor


# Default hyperparameters — chosen for a reasonable balance between
# accuracy and training speed on ~3 years of hourly data (~26 k rows).
DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_leaf": 5,
    "min_samples_split": 10,
    "max_features": 0.7,
    "n_jobs": -1,
    "random_state": 42,
}


def train(X_train, y_train, X_val=None, y_val=None, **overrides):
    """
    Train a Random Forest regressor.

    Parameters
    ----------
    X_train, y_train : array-like
        Training features and target.
    X_val, y_val : array-like, optional
        Validation set (unused by RF, accepted for interface consistency).
    **overrides
        Any ``RandomForestRegressor`` kwargs to override defaults.

    Returns
    -------
    RandomForestRegressor
        The fitted model.
    """
    params = {**DEFAULT_PARAMS, **overrides}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model
