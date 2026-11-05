import uuid
import pytest
from chatbot.app import app, db, User, Scan

@pytest.fixture
def auth_client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_client() as client:
        with app.app_context():
            unique_email = f"ui_user_{uuid.uuid4().hex[:8]}@hospital.org"
            user = User(name="UI Test Clinician", email=unique_email, password="SecurePassword123!")
            db.session.add(user)
            db.session.commit()
            user_id = user.id

        with client.session_transaction() as sess:
            sess['user_id'] = user_id

        yield client, user_id

        with app.app_context():
            u = User.query.get(user_id)
            if u:
                Scan.query.filter_by(user_id=user_id).delete()
                db.session.delete(u)
                db.session.commit()

def test_login_page_renders():
    with app.test_client() as client:
        resp = client.get('/login')
        assert resp.status_code == 200
        html = resp.data.decode('utf-8')
        assert "Neuro" in html and "Scan" in html
        assert 'name="email"' in html
        assert 'name="password"' in html
        assert 'viewport' in html

def test_dashboard_renders(auth_client):
    client, _ = auth_client
    resp = client.get('/dashboard')
    assert resp.status_code == 200
    html = resp.data.decode('utf-8')
    assert "Dashboard" in html
    assert "Total Diagnostic Scans" in html
    assert "Non-Demented" in html
    assert "UI Test Clinician" in html

def test_index_analysis_renders(auth_client):
    client, _ = auth_client
    resp = client.get('/')
    assert resp.status_code == 200
    html = resp.data.decode('utf-8')
    assert "MRI Analysis" in html
    assert "drop-zone" in html
    assert "chat-input" in html

def test_reports_page_renders(auth_client):
    client, _ = auth_client
    resp = client.get('/reports')
    assert resp.status_code == 200
    html = resp.data.decode('utf-8')
    assert "Report" in html

def test_history_page_renders(auth_client):
    client, _ = auth_client
    resp = client.get('/history')
    assert resp.status_code == 200
    html = resp.data.decode('utf-8')
    assert "Scan History" in html or "History" in html

def test_chat_history_page_renders(auth_client):
    client, _ = auth_client
    resp = client.get('/chat_history')
    assert resp.status_code == 200
    html = resp.data.decode('utf-8')
    assert "Chat" in html
