import os
import uuid
import pytest
from chatbot.app import app, db, User

TEMPLATES = [
    "index.html",
    "dashboard.html",
    "login.html",
    "history.html",
    "reports.html",
    "chat_history.html"
]

@pytest.fixture
def auth_client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_client() as client:
        with app.app_context():
            unique_email = f"a11y_{uuid.uuid4().hex[:8]}@hospital.org"
            user = User(name="Accessibility Tester", email=unique_email, password="SecurePass123!")
            db.session.add(user)
            db.session.commit()
            uid = user.id
        with client.session_transaction() as sess:
            sess['user_id'] = uid
        yield client, uid
        with app.app_context():
            u = User.query.get(uid)
            if u:
                db.session.delete(u)
                db.session.commit()

def test_viewport_meta_tags():
    templates_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chatbot", "templates")
    for tpl_name in TEMPLATES:
        tpl_path = os.path.join(templates_dir, tpl_name)
        assert os.path.exists(tpl_path), f"Missing template: {tpl_name}"
        with open(tpl_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert 'name="viewport"' in content, f"Template {tpl_name} missing responsive viewport meta tag"
        assert "width=device-width" in content, f"Template {tpl_name} missing device-width viewport setting"

def test_aria_landmarks_in_analysis_view(auth_client):
    client, _ = auth_client
    resp = client.get('/')
    assert resp.status_code == 200
    html = resp.data.decode('utf-8')
    assert 'role="main"' in html
    assert 'aria-label="MRI Scan Upload and Results"' in html
    assert 'aria-label="Interactive Clinical Chat Assistant"' in html
    assert 'aria-live="polite"' in html
    assert 'role="alert"' in html

def test_keyboard_focusable_dropzone(auth_client):
    client, _ = auth_client
    resp = client.get('/')
    assert resp.status_code == 200
    html = resp.data.decode('utf-8')
    assert 'tabindex="0"' in html
    assert 'role="button"' in html

def test_dashboard_responsive_table_container(auth_client):
    client, _ = auth_client
    resp = client.get('/dashboard')
    assert resp.status_code == 200
    html = resp.data.decode('utf-8')
    # Table must either be empty-state or wrapped in responsive overflow
    assert 'table-responsive' in html or 'empty-state' in html
