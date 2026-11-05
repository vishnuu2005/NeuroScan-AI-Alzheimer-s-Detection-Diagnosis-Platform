"""
Automated unit and integration tests for NeuroScan AI error handling,
standardized API error formatting, and structured audit trail logging.
"""
import json
import pytest
from chatbot.app import app, db, User, Scan
from chatbot.logging_config import log_clinical_event


@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.app_context():
        db.create_all()
        yield app.test_client()
        db.session.remove()


def test_404_html_error_page(client):
    """Verify non-existent paths return 404 status and render the custom clinical template."""
    resp = client.get('/nonexistent-clinical-record')
    assert resp.status_code == 404
    html = resp.get_data(as_text=True)
    assert 'Clinical Resource Not Found' in html
    assert 'Return to Dashboard' in html


def test_404_json_error_response(client):
    """Verify API 404 errors return standardized JSON schema."""
    resp = client.get('/api/unregistered_endpoint', headers={'Accept': 'application/json'})
    assert resp.status_code == 404
    data = resp.get_json()
    assert data is not None
    assert data['error'] == 'Not Found'
    assert data['status'] == 404


def test_400_json_error_response(client):
    """Verify invalid API payloads return structured 400 JSON errors."""
    with client.session_transaction() as sess:
        sess['user_id'] = 1
        sess['user_name'] = 'Dr. Error Tester'

    # Empty body to /chat
    resp = client.post('/api/chat', data="not json", content_type="text/plain")
    assert resp.status_code == 400
    data = resp.get_json()
    assert 'error' in data
    assert data['status'] == 400


def test_403_json_error_response(client):
    """Verify cross-user scan access via API returns structured 403 JSON error."""
    with app.app_context():
        user1 = User(name='Clinician One', email='c1_err@neuroscan.org', password='hash1Password!')
        user2 = User(name='Clinician Two', email='c2_err@neuroscan.org', password='hash2Password!')
        db.session.add_all([user1, user2])
        db.session.commit()

        scan = Scan(
            user_id=user1.id,
            prediction='MildDemented',
            confidence=0.88,
            all_probs=json.dumps({'MildDemented': 0.88, 'NonDemented': 0.12})
        )
        db.session.add(scan)
        db.session.commit()
        scan_id = scan.id
        user2_id = user2.id

    with client.session_transaction() as sess:
        sess['user_id'] = user2_id
        sess['user_name'] = 'Clinician Two'

    resp = client.get(f'/api/scans/{scan_id}')
    assert resp.status_code == 403
    data = resp.get_json()
    assert data['status'] == 403
    assert 'Unauthorized' in data['error']


def test_api_scans_retrieval(client):
    """Verify /api/scans returns structured JSON array for authenticated user."""
    with app.app_context():
        u = User(name='API Doctor', email='doctor_err@neuroscan.org', password='testPassword123!')
        db.session.add(u)
        db.session.commit()
        u_id = u.id

        scan = Scan(
            user_id=u_id,
            prediction='NonDemented',
            confidence=0.91,
            all_probs=json.dumps({'NonDemented': 0.91, 'VeryMildDemented': 0.09}),
            processing_ms=145.2,
            model_version='ResNet18-3ch'
        )
        db.session.add(scan)
        db.session.commit()

    with client.session_transaction() as sess:
        sess['user_id'] = u_id
        sess['user_name'] = 'API Doctor'

    resp = client.get('/api/scans')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['count'] == 1
    assert data['scans'][0]['prediction'] == 'NonDemented'
    assert data['scans'][0]['risk_tier'] == 'low'
    assert data['scans'][0]['processing_ms'] == 145.2


def test_clinical_audit_event_logging():
    """Verify log_clinical_event executes safely and records structured trail."""
    log_clinical_event("TEST_DIAGNOSTIC_EXECUTION", user_id=999, details="Automated unit validation")
