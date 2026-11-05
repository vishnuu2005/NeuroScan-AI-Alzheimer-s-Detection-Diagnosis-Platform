import io
import json
import uuid
import pytest
from PIL import Image
from chatbot.app import app, db, User, Scan, ChatSession, ChatMessage

def create_synthetic_mri():
    buf = io.BytesIO()
    img = Image.new("L", (224, 224), color=110)
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def test_full_user_journey_registration_to_prediction():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    client = app.test_client()

    email = f"journey_{uuid.uuid4().hex[:8]}@hospital.org"
    password = "StrongPassword2026!"

    # 1. Register
    reg_resp = client.post('/register', data={
        'name': 'Dr. Journey Tester',
        'email': email,
        'password': password
    }, follow_redirects=True)
    assert reg_resp.status_code == 200

    # 2. Login
    login_resp = client.post('/login', data={
        'email': email,
        'password': password
    }, follow_redirects=True)
    assert login_resp.status_code == 200

    # 3. Upload MRI Scan
    mri_data = {'file': (create_synthetic_mri(), 'patient_axial.png')}
    upload_resp = client.post('/', data=mri_data, content_type='multipart/form-data', follow_redirects=True)
    assert upload_resp.status_code == 200
    html = upload_resp.data.decode('utf-8')

    # 4. Check results displayed
    assert "Predicted Classification" in html
    assert "Confidence Level" in html
    assert "chart-data" in html

    # 5. Check database
    with app.app_context():
        user = User.query.filter_by(email=email).first()
        assert user is not None
        assert len(user.scans) >= 1
        scan = user.scans[0]
        assert scan.prediction in ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented", "Uncertain"]
        user_id = user.id

    # 6. Chat inquiry in scan context
    chat_resp = client.post('/chat', json={'message': 'What clinical steps should follow this classification?'})
    assert chat_resp.status_code == 200
    chat_data = chat_resp.get_json()
    assert chat_data['status'] == 'success'
    assert len(chat_data['response']) > 0

    # 7. Download PDF report
    report_resp = client.get('/download_report')
    assert report_resp.status_code == 200
    assert report_resp.content_type == 'application/pdf'
    assert report_resp.data.startswith(b'%PDF-')

    # 8. View Dashboard and Scan History
    dash_resp = client.get('/dashboard')
    assert dash_resp.status_code == 200
    assert "Recent Scans" in dash_resp.data.decode('utf-8')

    hist_resp = client.get('/history')
    assert hist_resp.status_code == 200

    # 9. Logout
    logout_resp = client.get('/logout', follow_redirects=True)
    assert logout_resp.status_code == 200
    assert "Sign In" in logout_resp.data.decode('utf-8')

    # Cleanup
    with app.app_context():
        u = User.query.get(user_id)
        if u:
            Scan.query.filter_by(user_id=user_id).delete()
            for cs in ChatSession.query.filter_by(user_id=user_id).all():
                ChatMessage.query.filter_by(chat_session_id=cs.id).delete()
                db.session.delete(cs)
            db.session.delete(u)
            db.session.commit()

def test_persistent_chat_consultation_context():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    client = app.test_client()

    with app.app_context():
        email = f"chatflow_{uuid.uuid4().hex[:8]}@hospital.org"
        user = User(name="Chatflow Tester", email=email, password="Password123!")
        db.session.add(user)
        db.session.commit()
        uid = user.id

    with client.session_transaction() as sess:
        sess['user_id'] = uid
        sess['stage'] = "MildDemented"
        sess['confidence'] = 78.5

    # Turn 1
    r1 = client.post('/chat', json={'message': 'What are typical symptoms?'})
    assert r1.status_code == 200
    assert "Symptom" in r1.get_json()['response']

    # Turn 2
    r2 = client.post('/chat', json={'message': 'What treatments or therapies are suggested?'})
    assert r2.status_code == 200
    assert "Therapeutic" in r2.get_json()['response'] or "Cholinesterase" in r2.get_json()['response']

    # Verify 4 messages in DB (2 user, 2 assistant)
    with app.app_context():
        cs = ChatSession.query.filter_by(user_id=uid).first()
        assert cs is not None
        assert len(cs.messages) == 4
        
        # Cleanup
        ChatMessage.query.filter_by(chat_session_id=cs.id).delete()
        db.session.delete(cs)
        u = User.query.get(uid)
        if u:
            db.session.delete(u)
        db.session.commit()

def test_pdf_report_caching_and_download():
    app.config['TESTING'] = True
    client = app.test_client()
    with app.app_context():
        email = f"pdfjourney_{uuid.uuid4().hex[:8]}@clinic.org"
        user = User(name="PDF Tester", email=email, password="Password123!")
        db.session.add(user)
        db.session.commit()
        uid = user.id

        scan = Scan(
            user_id=uid,
            prediction="VeryMildDemented",
            confidence=81.2,
            all_probs=json.dumps({"VeryMildDemented": 81.2, "NonDemented": 18.8})
        )
        db.session.add(scan)
        db.session.commit()
        scan_id = scan.id

    with client.session_transaction() as sess:
        sess['user_id'] = uid

    resp = client.get(f'/download_report/{scan_id}')
    assert resp.status_code == 200
    assert resp.content_type == 'application/pdf'
    assert resp.data.startswith(b'%PDF-')

    # Cleanup
    with app.app_context():
        Scan.query.filter_by(user_id=uid).delete()
        u = User.query.get(uid)
        if u:
            db.session.delete(u)
        db.session.commit()

def test_history_pagination_and_deletion_flow():
    app.config['TESTING'] = True
    client = app.test_client()
    with app.app_context():
        email = f"histflow_{uuid.uuid4().hex[:8]}@clinic.org"
        user = User(name="History Tester", email=email, password="Password123!")
        db.session.add(user)
        db.session.commit()
        uid = user.id

        scans = [
            Scan(user_id=uid, prediction="NonDemented", confidence=92.0, all_probs=json.dumps({"NonDemented": 92.0})),
            Scan(user_id=uid, prediction="ModerateDemented", confidence=84.0, all_probs=json.dumps({"ModerateDemented": 84.0}))
        ]
        db.session.add_all(scans)
        db.session.commit()
        s1_id = scans[0].id

    with client.session_transaction() as sess:
        sess['user_id'] = uid

    # Check history view
    h_resp = client.get('/history')
    assert h_resp.status_code == 200

    # Delete first scan
    del_resp = client.post(f'/delete_scan/{s1_id}')
    assert del_resp.status_code == 200

    with app.app_context():
        remaining = Scan.query.filter_by(user_id=uid).all()
        assert len(remaining) == 1
        Scan.query.filter_by(user_id=uid).delete()
        u = User.query.get(uid)
        if u:
            db.session.delete(u)
        db.session.commit()

def test_logout_and_unauthorized_redirection():
    app.config['TESTING'] = True
    client = app.test_client()

    with app.app_context():
        email = f"logout_{uuid.uuid4().hex[:8]}@clinic.org"
        user = User(name="Logout Tester", email=email, password="Password123!")
        db.session.add(user)
        db.session.commit()
        uid = user.id

    with client.session_transaction() as sess:
        sess['user_id'] = uid

    # Logout
    l_resp = client.get('/logout')
    assert l_resp.status_code == 302
    assert '/login' in l_resp.headers.get('Location', '')

    # Protected routes redirect to login
    for route in ['/', '/dashboard', '/reports', '/history', '/chat_history']:
        r = client.get(route)
        assert r.status_code == 302
        assert '/login' in r.headers.get('Location', '')

    with app.app_context():
        u = User.query.get(uid)
        if u:
            db.session.delete(u)
        db.session.commit()
