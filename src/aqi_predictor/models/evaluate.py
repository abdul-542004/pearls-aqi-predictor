"""
Model evaluation utilities.

Provides consistent metrics (RMSE, MAE, R²) for all models
and a comparison table for the training pipeline summary.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_true, y_pred):
    """
    Compute regression metrics.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted values.

    Returns
    -------
    dict
        ``{"rmse": float, "mae": float, "r2": float}``
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def print_metrics(metrics, model_name="Model"):
    """Pretty-print a metrics dict."""
    print(f"\n  {model_name}")
    print(f"    RMSE : {metrics['rmse']:.4f}")
    print(f"    MAE  : {metrics['mae']:.4f}")
    print(f"    R2   : {metrics['r2']:.4f}")


def compare_models(results):
    """
    Build a comparison DataFrame and identify the best model.

    Parameters
    ----------
    results : dict[str, dict]
        Mapping of ``{model_name: metrics_dict}``.

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by RMSE (ascending).
    str
        Name of the best model (lowest RMSE).
    """
    df = pd.DataFrame(results).T
    df.index.name = "model"
    df = df.sort_values("rmse")

    print("\n================================================")
    print("           Model Comparison (Test Set)           ")
    print("================================================")
    print(df.to_string())
    print("================================================")

    best_name = df.index[0]
    print(f"\n  * Best model: {best_name} (RMSE={df.loc[best_name, 'rmse']:.4f})")

    return df, best_name
