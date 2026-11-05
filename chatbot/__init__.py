"""
NeuroScan AI — Alzheimer's MRI Detection & Clinical Diagnosis Platform.

Version: 2.0.0-stable
Author: vishnuu2005
Repository: https://github.com/vishnuu2005/NeuroScan-AI-Alzheimer-s-Detection-Diagnosis-Platform
"""

__title__ = "NeuroScan AI"
__version__ = "2.0.0"
__author__ = "vishnuu2005"
__license__ = "MIT"
__status__ = "Production/Stable"

from .app import app, db
from .inference import load_model, predict, transform_image, warmup_model
from .generate_reports import generate_report, generate_report_stream
from .validators import validate_mri_image, allowed_file
from .formatters import get_risk_tier, format_stage_label, prepare_chart_dataset

__all__ = [
    "app",
    "db",
    "load_model",
    "predict",
    "transform_image",
    "warmup_model",
    "generate_report",
    "generate_report_stream",
    "validate_mri_image",
    "allowed_file",
    "get_risk_tier",
    "format_stage_label",
    "prepare_chart_dataset",
]
