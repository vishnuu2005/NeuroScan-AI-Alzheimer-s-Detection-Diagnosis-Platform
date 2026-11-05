import pytest
from chatbot.app import app, db, User

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client

def test_login_page_renders(client):
    res = client.get('/login')
    assert res.status_code == 200
    assert b"Sign In" in res.data or b"Login" in res.data or b"password" in res.data

def test_register_invalid_email(client):
    res = client.post('/register', data={
        'name': 'Clinician User',
        'email': 'not-an-email',
        'password': 'ValidPassword123!'
    }, follow_redirects=True)
    assert res.status_code == 200
    assert b"valid clinical email" in res.data or b"error" in res.data

def test_register_short_password(client):
    res = client.post('/register', data={
        'name': 'Clinician User',
        'email': 'doc@hospital.org',
        'password': 'short'
    }, follow_redirects=True)
    assert res.status_code == 200
    assert b"at least 8 characters" in res.data

def test_register_and_login_success(client):
    email = "neurologist_test@hospital.org"
    password = "StrongPassword2026!"
    
    # Register
    res = client.post('/register', data={
        'name': 'Dr. Test Neurologist',
        'email': email,
        'password': password
    }, follow_redirects=True)
    assert res.status_code == 200
    
    # Login
    res = client.post('/login', data={
        'email': email,
        'password': password
    }, follow_redirects=True)
    assert res.status_code == 200
    assert res.request.path == "/dashboard"

def test_protected_routes_redirect_unauthenticated(client):
    # Unauthenticated GET / should redirect to login
    res = client.get('/', follow_redirects=False)
    assert res.status_code == 302
    assert "/login" in res.headers["Location"]
    
    # Unauthenticated GET /dashboard should redirect to login
    res = client.get('/dashboard', follow_redirects=False)
    assert res.status_code == 302
    assert "/login" in res.headers["Location"]
