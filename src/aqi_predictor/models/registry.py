"""
Model registry — local persistence and Hopsworks upload.

Handles two serialisation formats:
  * sklearn / xgboost → ``joblib``
  * PyTorch (LSTM)    → ``torch.save`` (state_dict)
"""

import json
import os
import tempfile
from pathlib import Path

import hopsworks
import joblib


# ── Local persistence ───────────────────────────────────────────────


def save_model_local(model, model_name, metrics, artifacts_dir, extra_files=None):
    """
    Save a model and its metrics to ``<artifacts_dir>/<model_name>/``.

    Parameters
    ----------
    model : object
        Fitted model (sklearn, xgboost, or PyTorch).
    model_name : str
        Subdirectory name, e.g. ``"random_forest"``.
    metrics : dict
        ``{"rmse": …, "mae": …, "r2": …}``
    artifacts_dir : str | Path
        Root artifacts directory (``artifacts/models``).
    extra_files : dict[str, object] | None
        Additional objects to save via joblib (e.g. ``{"scaler": scaler}``).

    Returns
    -------
    Path
        Directory where the model was saved.
    """
    model_dir = Path(artifacts_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    _save_model_file(model, model_dir)

    # Save extra artefacts (e.g. scaler for LSTM)
    if extra_files:
        for name, obj in extra_files.items():
            joblib.dump(obj, model_dir / f"{name}.pkl")

    # Save metrics
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved {model_name} -> {model_dir}")
    return model_dir


def _save_model_file(model, model_dir):
    """Dispatch to the right serialiser based on model type."""
    try:
        import torch.nn as nn

        if isinstance(model, nn.Module):
            import torch

            torch.save(model.state_dict(), model_dir / "model.pt")
            return
    except ImportError:
        pass

    # sklearn / xgboost
    joblib.dump(model, model_dir / "model.pkl")


# ── Hopsworks upload ────────────────────────────────────────────────


def upload_to_hopsworks(model_dir, model_name, metrics):
    """
    Upload a model directory to the Hopsworks Model Registry.

    Parameters
    ----------
    model_dir : str | Path
        Local directory containing the model file(s) and metrics.json.
    model_name : str
        Name under which to register the model in Hopsworks.
    metrics : dict
        Metrics to attach to the registry entry.
    """
    model_dir = Path(model_dir)

    # Ensure /tmp exists on Windows for Hopsworks internal cert/PEM files
    try:
        os.makedirs("/tmp", exist_ok=True)
    except Exception:
        pass

    cert_folder = os.environ.get(
        "HOPSWORKS_CERT_FOLDER",
        os.path.join(tempfile.gettempdir(), "hopsworks_certs"),
    )
    os.makedirs(cert_folder, exist_ok=True)

    project = hopsworks.login(
        api_key_value=os.environ.get("HOPSWORKS_API_KEY"),
        project=os.environ.get("HOPSWORKS_PROJECT"),
        cert_folder=cert_folder,
    )
    mr = project.get_model_registry()

    # Create or version-bump the model
    hw_model = mr.python.create_model(
        name=model_name,
        metrics=metrics,
        description=f"AQI forecaster — {model_name}",
    )

    hw_model.save(str(model_dir), keep_original_files=True)
    print(f"  Uploaded {model_name} to Hopsworks Model Registry (version {hw_model.version})")
