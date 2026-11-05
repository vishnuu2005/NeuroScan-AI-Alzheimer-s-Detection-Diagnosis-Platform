"""
Automated unit tests for system diagnostic and health check endpoints.
Verifies /health and /api/health response contracts and database monitoring.
"""
import pytest
from chatbot.app import app, db


@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['WTF_CSRF_ENABLED'] = False
    with app.app_context():
        db.create_all()
        yield app.test_client()
        db.session.remove()
        db.drop_all()


def test_health_endpoint_contract(client):
    """Verify /health returns 200, healthy status, and diagnostic metadata."""
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data is not None
    assert data['status'] == 'healthy'
    assert data['database'] == 'ok'
    assert 'model' in data
    assert 'version' in data
    assert data['version'] == '2.0.0'


def test_api_health_endpoint_alias(client):
    """Verify /api/health returns identical JSON schema with security headers."""
    resp = client.get('/api/health')
    assert resp.status_code == 200
    assert resp.headers.get('X-Content-Type-Options') == 'nosniff'
    data = resp.get_json()
    assert data['status'] == 'healthy'
    assert 'classes' in data['model']
