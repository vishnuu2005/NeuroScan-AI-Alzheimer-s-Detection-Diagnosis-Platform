"""
Structured logging and audit trail configuration for NeuroScan AI.
Sets up rotating file handlers and console loggers for tracking
clinical diagnostic events, user authentication, and system errors.
"""
import os
import logging
from logging.handlers import RotatingFileHandler


def configure_logging(app=None, log_dir=None):
    """
    Configures structured logging with rotating file handler and standard formatting.
    """
    if log_dir is None:
        if app is not None:
            base_dir = os.path.dirname(os.path.abspath(app.root_path))
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(base_dir, 'logs')

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'neuroscan.log')
    audit_file = os.path.join(log_dir, 'clinical_audit.log')

    # Main rotating log handler (10 MB per file, max 5 backups)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    formatter = logging.Formatter(
        '[%(asctime)s] [%(process)d] [%(levelname)s] [%(name)s] %(message)s [in %(pathname)s:%(lineno)d]'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Dedicated clinical audit logger
    audit_handler = RotatingFileHandler(
        audit_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
        encoding='utf-8'
    )
    audit_formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "event": "CLINICAL_AUDIT", "level": "%(levelname)s", %(message)s}'
    )
    audit_handler.setFormatter(audit_formatter)
    audit_handler.setLevel(logging.INFO)

    audit_logger = logging.getLogger('neuroscan.audit')
    audit_logger.setLevel(logging.INFO)
    if not audit_logger.handlers:
        audit_logger.addHandler(audit_handler)

    if app is not None:
        app.logger.setLevel(logging.INFO)
        if not any(isinstance(h, RotatingFileHandler) for h in app.logger.handlers):
            app.logger.addHandler(file_handler)
        app.audit_logger = audit_logger

    return file_handler, audit_logger


def log_clinical_event(event_type, user_id=None, details=None):
    """
    Convenience helper to record HIPAA/audit-conscious diagnostic trail events.
    """
    audit_logger = logging.getLogger('neuroscan.audit')
    user_str = f'"user_id": {user_id}' if user_id is not None else '"user_id": "anonymous"'
    details_str = f', "details": "{details}"' if details else ''
    audit_logger.info(f'"event_type": "{event_type}", {user_str}{details_str}')
