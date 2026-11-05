import os
import json
import uuid
import pytest
from chatbot.app import app, db, User, Scan
from chatbot.generate_reports import generate_report

@pytest.fixture
def auth_client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_client() as client:
        with app.app_context():
            unique_email = f"report_user_{uuid.uuid4().hex[:8]}@hospital.org"
            user = User(name="Report Tester", email=unique_email, password="SecurePassword123!")
            db.session.add(user)
            db.session.commit()
            user_id = user.id

        with client.session_transaction() as sess:
            sess['user_id'] = user_id
            sess['stage'] = "MildDemented"
            sess['confidence'] = 76.4
            sess['all_probs'] = {
                "NonDemented": 5.1,
                "VeryMildDemented": 12.3,
                "MildDemented": 76.4,
                "ModerateDemented": 6.2
            }

        yield client, user_id

        with app.app_context():
            u = User.query.get(user_id)
            if u:
                Scan.query.filter_by(user_id=user_id).delete()
                db.session.delete(u)
                db.session.commit()

def test_generate_report_creates_valid_pdf(tmp_path):
    out_file = str(tmp_path / "test_report.pdf")
    sample_probs = {
        "NonDemented": 10.0,
        "VeryMildDemented": 15.0,
        "MildDemented": 70.0,
        "ModerateDemented": 5.0
    }
    result_path = generate_report(
        prediction="MildDemented",
        confidence=70.0,
        all_probs=sample_probs,
        output_path=out_file,
        model_version="ResNet18-3ch",
        reference_id="NS-TEST-001"
    )
    assert os.path.exists(result_path)
    assert os.path.getsize(result_path) > 5000
    with open(result_path, "rb") as f:
        header = f.read(5)
        assert header == b"%PDF-"

def test_generate_report_uncertain_classification(tmp_path):
    out_file = str(tmp_path / "uncertain_report.pdf")
    sample_probs = {
        "NonDemented": 25.0,
        "VeryMildDemented": 25.0,
        "MildDemented": 25.0,
        "ModerateDemented": 25.0
    }
    result_path = generate_report(
        prediction="Uncertain",
        confidence=25.0,
        all_probs=sample_probs,
        output_path=out_file
    )
    assert os.path.exists(result_path)
    with open(result_path, "rb") as f:
        assert f.read(5) == b"%PDF-"

def test_download_report_unauthenticated_blocked():
    with app.test_client() as client:
        resp = client.get('/download_report')
        assert resp.status_code == 302
        assert '/login' in resp.headers.get('Location', '')

def test_download_report_no_session_prediction():
    with app.test_client() as client:
        with app.app_context():
            user = User(name="No Scan User", email=f"noscan_{uuid.uuid4().hex[:8]}@test.org", password="Password123!")
            db.session.add(user)
            db.session.commit()
            uid = user.id
        with client.session_transaction() as sess:
            sess['user_id'] = uid

        resp = client.get('/download_report')
        assert resp.status_code == 400
        assert b"No prediction found" in resp.data

        with app.app_context():
            u = User.query.get(uid)
            if u:
                db.session.delete(u)
                db.session.commit()

def test_download_report_authenticated_success(auth_client):
    client, _ = auth_client
    resp = client.get('/download_report')
    assert resp.status_code == 200
    assert resp.content_type == 'application/pdf'
    assert resp.data.startswith(b'%PDF-')

def test_download_report_by_id_authorization(auth_client):
    client, user_id = auth_client
    with app.app_context():
        # Create scan for user
        scan = Scan(
            user_id=user_id,
            prediction="NonDemented",
            confidence=94.2,
            all_probs=json.dumps({"NonDemented": 94.2, "MildDemented": 5.8})
        )
        db.session.add(scan)
        
        # Create other user and scan
        other_user = User(name="Other Clinician", email=f"other_{uuid.uuid4().hex[:8]}@clinic.org", password="Pass123!")
        db.session.add(other_user)
        db.session.commit()
        
        other_scan = Scan(
            user_id=other_user.id,
            prediction="ModerateDemented",
            confidence=82.1,
            all_probs=json.dumps({"ModerateDemented": 82.1})
        )
        db.session.add(other_scan)
        db.session.commit()
        
        own_scan_id = scan.id
        other_scan_id = other_scan.id
        other_user_id = other_user.id

    # Downloading own scan should succeed
    resp_own = client.get(f'/download_report/{own_scan_id}')
    assert resp_own.status_code == 200
    assert resp_own.data.startswith(b'%PDF-')

    # Downloading other user's scan should return 403 Forbidden
    resp_other = client.get(f'/download_report/{other_scan_id}')
    assert resp_other.status_code == 403

    # Cleanup
    with app.app_context():
        Scan.query.filter(Scan.id.in_([own_scan_id, other_scan_id])).delete()
        u2 = User.query.get(other_user_id)
        if u2:
            db.session.delete(u2)
        db.session.commit()
