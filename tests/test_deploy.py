"""
Deployment verification and CI test suite for NeuroScan AI.
Validates the WSGI entrypoint, container configuration, Dockerfile directives,
and environment loading mechanics.
"""
import os
import sys
import pytest
import wsgi


def test_wsgi_application_callable():
    """Verify wsgi exposes standard WSGI application callable."""
    assert hasattr(wsgi, "application")
    assert callable(wsgi.application)
    assert hasattr(wsgi.application, "wsgi_app")


def test_wsgi_argument_parser(monkeypatch):
    """Verify parse_args parses default host, port, and thread constraints."""
    monkeypatch.setattr(sys, "argv", ["wsgi.py", "--host=0.0.0.0", "--port=8080", "--threads=8"])
    args = wsgi.parse_args()
    assert args.host == "0.0.0.0"
    assert args.port == 8080
    assert args.threads == 8


def test_dockerfile_syntax_and_directives():
    """Verify Dockerfile presence and essential clinical container directives."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dockerfile_path = os.path.join(base_dir, "Dockerfile")
    assert os.path.exists(dockerfile_path)
    
    with open(dockerfile_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "FROM python:3.11-slim" in content
    assert "useradd -r -g neuroscan" in content  # Non-root security
    assert "HEALTHCHECK" in content
    assert "EXPOSE 5000" in content


def test_docker_compose_configuration():
    """Verify docker-compose.yml volume mappings and healthcheck specification."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    compose_path = os.path.join(base_dir, "docker-compose.yml")
    assert os.path.exists(compose_path)

    with open(compose_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "neuroscan_platform" in content
    assert "5000:5000" in content
    assert "neuroscan_instance:" in content


def test_ci_workflow_specification():
    """Verify GitHub Actions CI workflow configuration."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ci_path = os.path.join(base_dir, ".github", "workflows", "ci.yml")
    assert os.path.exists(ci_path)

    with open(ci_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "NeuroScan AI CI" in content
    assert "windows-latest" in content
    assert "pytest" in content
