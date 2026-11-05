"""
Comprehensive automated security and penetration test suite for NeuroScan AI.
Tests OWASP security headers, directory traversal defenses, session cookie flags,
password policy enforcement, SQL injection immunity, and timing-attack mitigations.
"""
import pytest
from chatbot.app import app, db, User, DUMMY_PASSWORD_HASH
from werkzeug.security import check_password_hash


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


def test_security_headers_present(client):
    """Verify OWASP recommended security headers on HTTP responses."""
    resp = client.get('/login')
    assert resp.status_code == 200
    headers = resp.headers
    
    assert headers.get('X-Content-Type-Options') == 'nosniff'
    assert headers.get('X-Frame-Options') == 'SAMEORIGIN'
    assert headers.get('X-XSS-Protection') == '1; mode=block'
    assert 'Content-Security-Policy' in headers
    assert "default-src 'self'" in headers['Content-Security-Policy']
    assert headers.get('Referrer-Policy') == 'strict-origin-when-cross-origin'
    assert 'geolocation=()' in headers.get('Permissions-Policy', '')


def test_path_traversal_uploads_blocked(client):
    """Verify directory traversal in upload routing returns 404/400 and does not leak files."""
    with client.session_transaction() as sess:
        sess['user_id'] = 1
        sess['user_name'] = 'Dr. Security'

    traversal_payloads = [
        '../app.py',
        '..\\app.py',
        '....//app.py',
        '%2e%2e%2fapp.py',
        'sub/../../app.py'
    ]
    for payload in traversal_payloads:
        resp = client.get(f'/uploads/{payload}')
        assert resp.status_code in (400, 404)


def test_path_traversal_reports_blocked(client):
    """Verify directory traversal in report downloads is strictly denied."""
    with client.session_transaction() as sess:
        sess['user_id'] = 1
        sess['user_name'] = 'Dr. Security'

    traversal_payloads = [
        '../database.db',
        '..\\database.db',
        '....//test.txt'
    ]
    for payload in traversal_payloads:
        resp = client.get(f'/reports/{payload}')
        assert resp.status_code in (400, 404)


def test_password_length_boundary_enforcement(client):
    """Verify password policy boundaries (min 8 chars, max 128 chars)."""
    # Short password (< 8 chars)
    resp_short = client.post('/register', data={
        'name': 'Short Tester',
        'email': 'short@neuroscan.org',
        'password': 'short'
    }, follow_redirects=False)
    assert resp_short.status_code == 302
    with app.app_context():
        assert User.query.filter_by(email='short@neuroscan.org').first() is None

    # Excessively long password (> 128 chars) to prevent DoS via bcrypt/scrypt work factor
    long_pass = 'A' * 130
    resp_long = client.post('/register', data={
        'name': 'Long Tester',
        'email': 'long@neuroscan.org',
        'password': long_pass
    }, follow_redirects=False)
    assert resp_long.status_code == 302
    with app.app_context():
        assert User.query.filter_by(email='long@neuroscan.org').first() is None


def test_session_cookie_security_attributes(client):
    """Verify session cookies enforce HttpOnly and SameSite attributes."""
    client.post('/register', data={
        'name': 'Cookie Security',
        'email': 'cookie@neuroscan.org',
        'password': 'SecurePassword2026!'
    })
    resp = client.post('/login', data={
        'email': 'cookie@neuroscan.org',
        'password': 'SecurePassword2026!'
    })
    cookie_header = resp.headers.get('Set-Cookie', '')
    assert 'HttpOnly' in cookie_header
    assert 'SameSite=Lax' in cookie_header


def test_sql_injection_resilience(client):
    """Verify authentication queries use parameterized statements and resist SQL injection."""
    sqli_payloads = [
        "' OR '1'='1",
        "admin'--",
        "' UNION SELECT null, null, null--",
        "'; DROP TABLE user;--"
    ]
    for sqli in sqli_payloads:
        resp = client.post('/login', data={
            'email': sqli,
            'password': 'password123'
        })
        # Should cleanly refuse login without 500 error or SQL crash
        assert resp.status_code in (200, 400)
        with client.session_transaction() as sess:
            assert 'user_id' not in sess


def test_timing_attack_mitigation_token():
    """Verify DUMMY_PASSWORD_HASH exists and can be verified to prevent user enumeration."""
    assert DUMMY_PASSWORD_HASH is not None
    assert DUMMY_PASSWORD_HASH.startswith('scrypt:') or DUMMY_PASSWORD_HASH.startswith('pbkdf2:')
    assert check_password_hash(DUMMY_PASSWORD_HASH, "incorrect_probe_password") is False
