"""
Master Regression Test Suite for NeuroScan AI.
Executes an end-to-end clinical workflow through all platform layers:
System Health -> Auth -> Upload -> Inference -> Consultation -> Report Streaming -> Audit Logging -> Session Termination.
"""
import io
import json
import pytest
from PIL import Image
from chatbot.app import app, db, User, Scan, ChatSession
from chatbot.logging_config import log_clinical_event


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


def _create_mock_mri_png():
    """Generates an in-memory 224x224 grayscale PNG image simulating an MRI brain slice."""
    img = Image.new('L', (224, 224), color=128)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


def test_full_clinical_platform_regression(client):
    """
    Complete end-to-end regression verifying all subsystems work harmoniously:
    1. /health diagnostic verification
    2. Clinician registration and authentication
    3. MRI scan upload and ResNet inference
    4. Multi-class prediction verification and database persistence
    5. Interactive clinical chat consultation
    6. In-memory PDF report generation and download
    7. JSON REST API query for scan history
    8. Clinical event audit trail execution
    9. Session logout and unauthorized guard enforcement
    """
    # 1. Health check
    health_resp = client.get('/health')
    assert health_resp.status_code == 200
    assert health_resp.get_json()['status'] == 'healthy'

    # 2. Clinician Registration
    reg_resp = client.post('/register', data={
        'name': 'Dr. Marcus Vance',
        'email': 'm.vance@neuroscan-clinic.org',
        'password': 'ClinicalPassword2026!'
    }, follow_redirects=True)
    assert reg_resp.status_code == 200

    # 3. Authentication Login
    login_resp = client.post('/login', data={
        'email': 'm.vance@neuroscan-clinic.org',
        'password': 'ClinicalPassword2026!'
    }, follow_redirects=True)
    assert login_resp.status_code == 200
    assert b'Logout' in login_resp.data or b'Dashboard' in login_resp.data

    # 4. Upload MRI scan and execute inference
    mri_data = _create_mock_mri_png()
    upload_resp = client.post('/', data={
        'file': (mri_data, 'patient_axial_mri.png')
    }, content_type='multipart/form-data', follow_redirects=True)
    assert upload_resp.status_code == 200
    upload_html = upload_resp.get_data(as_text=True)
    assert 'Prediction Results' in upload_html or 'Diagnosis' in upload_html or 'Confidence' in upload_html

    # 5. Verify database scan record
    with app.app_context():
        user = User.query.filter_by(email='m.vance@neuroscan-clinic.org').first()
        assert user is not None
        user_id = user.id
        scans = Scan.query.filter_by(user_id=user.id).all()
        assert len(scans) == 1
        scan_id = scans[0].id
        assert scans[0].prediction in ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented", "Uncertain"]

    # 6. Interactive Clinical AI Chat
    chat_resp = client.post('/api/chat', json={
        'message': 'What are the recommended clinical guidelines for this patient?'
    })
    assert chat_resp.status_code == 200
    chat_json = chat_resp.get_json()
    assert chat_json['status'] == 'success'
    assert len(chat_json['response']) > 10

    # 7. In-memory PDF Report Download
    report_resp = client.get('/download_report')
    assert report_resp.status_code == 200
    assert report_resp.mimetype == 'application/pdf'
    assert report_resp.data.startswith(b'%PDF-')

    # 8. REST API scan history query
    api_resp = client.get('/api/scans')
    assert api_resp.status_code == 200
    api_data = api_resp.get_json()
    assert api_data['count'] == 1
    assert api_data['scans'][0]['id'] == scan_id

    # 9. Audit logging
    log_clinical_event("REGRESSION_VERIFICATION_COMPLETE", user_id=user_id, details="All 9 steps succeeded")

    # 10. Logout and route protection
    logout_resp = client.get('/logout', follow_redirects=True)
    assert logout_resp.status_code == 200

    # Unauthenticated query to dashboard must redirect to login
    dash_unauth = client.get('/dashboard')
    assert dash_unauth.status_code == 302
    assert '/login' in dash_unauth.headers.get('Location', '')
