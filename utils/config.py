"""
Configuration and Path Management for NeuroScan AI.
Provides centralized path resolution for data, models, uploads, and reports
across training, inference, and web service modules.
"""

import os
from pathlib import Path


def get_base_dir() -> Path:
    """Returns the root directory of the NeuroScan project."""
    env_base = os.getenv("NEUROSCAN_BASE_DIR")
    if env_base:
        return Path(env_base).resolve()
    # utils is one level below repository root
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Returns path to dataset root directory."""
    return Path(os.getenv("NEUROSCAN_DATA_DIR", str(get_base_dir() / "data")))


def get_model_dir() -> Path:
    """Returns path to neural network models directory."""
    model_dir = Path(os.getenv("NEUROSCAN_MODEL_DIR", str(get_base_dir() / "models")))
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_plot_dir() -> Path:
    """Returns path to training curve plots directory."""
    plot_dir = Path(os.getenv("NEUROSCAN_PLOT_DIR", str(get_base_dir() / "plots")))
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def get_upload_dir() -> Path:
    """Returns path to uploaded MRI scan storage."""
    upload_dir = Path(os.getenv("UPLOAD_FOLDER", str(get_base_dir() / "static" / "uploads")))
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def get_report_dir() -> Path:
    """Returns path to generated PDF reports."""
    report_dir = Path(os.getenv("REPORT_FOLDER", str(get_base_dir() / "static" / "reports")))
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir
