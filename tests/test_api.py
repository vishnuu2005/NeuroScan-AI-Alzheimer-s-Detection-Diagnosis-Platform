import json
import uuid
import pytest
from chatbot.app import app, db, User, Scan, ChatSession, ChatMessage

@pytest.fixture
def api_users():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.app_context():
        u1 = User(name="Clinician One", email=f"c1_{uuid.uuid4().hex[:8]}@hospital.org", password="HashedPassword1!")
        u2 = User(name="Clinician Two", email=f"c2_{uuid.uuid4().hex[:8]}@hospital.org", password="HashedPassword2!")
        db.session.add_all([u1, u2])
        db.session.commit()
        u1_id = u1.id
        u2_id = u2.id

    yield u1_id, u2_id

    with app.app_context():
        for uid in [u1_id, u2_id]:
            user = User.query.get(uid)
            if user:
                Scan.query.filter_by(user_id=uid).delete()
                for cs in ChatSession.query.filter_by(user_id=uid).all():
                    ChatMessage.query.filter_by(chat_session_id=cs.id).delete()
                    db.session.delete(cs)
                db.session.delete(user)
        db.session.commit()

def test_auth_lifecycle():
    client = app.test_client()
    unique_email = f"newuser_{uuid.uuid4().hex[:8]}@hospital.org"
    password = "ValidPassword123!"

    # 1. Registration
    reg_resp = client.post('/register', data={
        'name': 'New Clinical Staff',
        'email': unique_email,
        'password': password
    })
    assert reg_resp.status_code == 302
    assert '/login' in reg_resp.headers.get('Location', '')

    # 2. Login
    login_resp = client.post('/login', data={
        'email': unique_email,
        'password': password
    })
    assert login_resp.status_code == 302
    assert '/' in login_resp.headers.get('Location', '')

    # 3. Authenticated session access
    dash_resp = client.get('/dashboard')
    assert dash_resp.status_code == 200

    # 4. Logout
    logout_resp = client.get('/logout')
    assert logout_resp.status_code == 302
    assert '/login' in logout_resp.headers.get('Location', '')

    # 5. Accessing dashboard after logout redirects
    post_logout_resp = client.get('/dashboard')
    assert post_logout_resp.status_code == 302

    # Cleanup created user
    with app.app_context():
        User.query.filter_by(email=unique_email).delete()
        db.session.commit()

def test_scan_deletion_and_ownership(api_users):
    u1_id, u2_id = api_users
    client = app.test_client()

    with app.app_context():
        scan = Scan(
            user_id=u1_id,
            prediction="NonDemented",
            confidence=97.4,
            all_probs=json.dumps({"NonDemented": 97.4})
        )
        db.session.add(scan)
        db.session.commit()
        scan_id = scan.id

    # User 2 attempts to delete User 1's scan
    with client.session_transaction() as sess:
        sess['user_id'] = u2_id
    del_resp_unauthorized = client.post(f'/delete_scan/{scan_id}')
    assert del_resp_unauthorized.status_code == 403

    # User 1 deletes their own scan
    with client.session_transaction() as sess:
        sess['user_id'] = u1_id
    del_resp_authorized = client.post(f'/delete_scan/{scan_id}')
    assert del_resp_authorized.status_code == 200
    assert del_resp_authorized.get_json() == {'status': 'deleted'}

    # Verify scan is deleted from database
    with app.app_context():
        assert Scan.query.get(scan_id) is None

def test_scan_boundary_conditions(api_users):
    u1_id, _ = api_users
    client = app.test_client()
    with client.session_transaction() as sess:
        sess['user_id'] = u1_id

    # Non-existent scan returns 404
    resp_missing = client.post('/delete_scan/888888')
    assert resp_missing.status_code == 404

def test_chat_session_deletion_and_cascade(api_users):
    u1_id, u2_id = api_users
    client = app.test_client()

    with app.app_context():
        chat_sess = ChatSession(user_id=u1_id, scan_stage="MildDemented")
        db.session.add(chat_sess)
        db.session.commit()
        msg = ChatMessage(chat_session_id=chat_sess.id, role="user", content="Hello assistant")
        db.session.add(msg)
        db.session.commit()
        sess_id = chat_sess.id
        msg_id = msg.id

    # User 2 cannot delete User 1's chat
    with client.session_transaction() as sess:
        sess['user_id'] = u2_id
    resp_unauth = client.post(f'/delete_chat/{sess_id}')
    assert resp_unauth.status_code == 403

    # User 1 deletes own chat
    with client.session_transaction() as sess:
        sess['user_id'] = u1_id
    resp_auth = client.post(f'/delete_chat/{sess_id}')
    assert resp_auth.status_code == 200
    assert resp_auth.get_json() == {'status': 'deleted'}

    # Verify cascade deletion of both session and messages
    with app.app_context():
        assert ChatSession.query.get(sess_id) is None
        assert ChatMessage.query.get(msg_id) is None

def test_chat_session_not_found(api_users):
    u1_id, _ = api_users
    client = app.test_client()
    with client.session_transaction() as sess:
        sess['user_id'] = u1_id
    resp_missing = client.post('/delete_chat/777777')
    assert resp_missing.status_code == 404

def test_malformed_requests_and_unsupported_methods(api_users):
    u1_id, _ = api_users
    client = app.test_client()
    with client.session_transaction() as sess:
        sess['user_id'] = u1_id

    # 1. Non-JSON payload to /chat
    resp_non_json = client.post('/chat', data="plain text message", content_type="text/plain")
    assert resp_non_json.status_code == 400

    # 2. GET on POST-only endpoints
    resp_get_del_scan = client.get('/delete_scan/99999')
    assert resp_get_del_scan.status_code == 405

    resp_get_del_chat = client.get('/delete_chat/99999')
    assert resp_get_del_chat.status_code == 405

    # 3. Non-existent resource deletion returns 404
    resp_not_found = client.post('/delete_scan/999999')
    assert resp_not_found.status_code == 404

def test_unsupported_http_methods(api_users):
    u1_id, _ = api_users
    client = app.test_client()
    with client.session_transaction() as sess:
        sess['user_id'] = u1_id

    # GET on clear_chat returns 405
    resp_get_clear = client.get('/clear_chat')
    assert resp_get_clear.status_code == 405

    # PUT on login returns 405
    resp_put_login = client.put('/login')
    assert resp_put_login.status_code == 405
