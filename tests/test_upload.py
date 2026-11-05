import io
import os
import uuid
import pytest
from PIL import Image
from chatbot.app import app, db, User, Scan
from chatbot.validators import allowed_file, verify_magic_bytes, validate_mri_image, ALLOWED_EXTENSIONS

@pytest.fixture
def auth_client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_client() as client:
        with app.app_context():
            unique_email = f"tester_{uuid.uuid4().hex[:8]}@hospital.org"
            user = User(name="Upload Tester", email=unique_email, password="SecurePass123!")
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

def create_synthetic_image(width=128, height=128, fmt="PNG"):
    buf = io.BytesIO()
    img = Image.new("L", (width, height), color=128)
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf

def test_allowed_file_whitelist():
    assert allowed_file("scan.png") is True
    assert allowed_file("mri.jpg") is True
    assert allowed_file("PATIENT_SCAN.JPEG") is True
    assert allowed_file("brain.webp") is True
    assert allowed_file("scan.tif") is True
    assert allowed_file("scan.tiff") is True
    assert allowed_file("exploit.exe") is False
    assert allowed_file("script.py") is False
    assert allowed_file("payload.php") is False
    assert allowed_file("no_extension") is False

def test_verify_magic_bytes():
    # Valid PNG
    png_buf = create_synthetic_image(fmt="PNG")
    valid, info = verify_magic_bytes(png_buf)
    assert valid is True
    assert info == "png"

    # Valid JPEG
    jpg_buf = create_synthetic_image(fmt="JPEG")
    valid, info = verify_magic_bytes(jpg_buf)
    assert valid is True
    assert info == "jpeg"

    # Spoofed file (plain text pretending to be image)
    fake_buf = io.BytesIO(b"MZ This is not an image file")
    valid, info = verify_magic_bytes(fake_buf)
    assert valid is False

def test_dimension_validation():
    # Tiny image under 64x64 threshold
    tiny_buf = create_synthetic_image(width=32, height=32, fmt="PNG")
    tiny_buf.filename = "tiny.png"
    tiny_buf.stream = tiny_buf
    valid, msg = validate_mri_image(tiny_buf)
    assert valid is False
    assert "too small" in msg

    # Valid dimension image
    valid_buf = create_synthetic_image(width=224, height=224, fmt="PNG")
    valid_buf.filename = "normal.png"
    valid_buf.stream = valid_buf
    valid, info = validate_mri_image(valid_buf)
    assert valid is True
    assert info["dimensions"] == (224, 224)

def test_unauthenticated_upload_redirect():
    with app.test_client() as client:
        buf = create_synthetic_image()
        data = {'file': (buf, 'scan.png')}
        resp = client.post('/', data=data, content_type='multipart/form-data')
        assert resp.status_code == 302
        assert '/login' in resp.headers.get('Location', '')

def test_authenticated_upload_valid_scan(auth_client):
    client, user_id = auth_client
    buf = create_synthetic_image(width=224, height=224, fmt="PNG")
    data = {'file': (buf, 'brain_axial.png')}
    resp = client.post('/', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert resp.status_code == 200
    
    with app.app_context():
        scans = Scan.query.filter_by(user_id=user_id).all()
        assert len(scans) >= 1
        assert scans[0].prediction in ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented", "Uncertain"]

def test_authenticated_upload_spoofed_content(auth_client):
    client, user_id = auth_client
    # Text payload disguised with .png filename
    buf = io.BytesIO(b"<?php echo 'malicious'; ?>")
    data = {'file': (buf, 'fake.png')}
    resp = client.post('/', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert resp.status_code == 200
    assert b"Security check failed" in resp.data or b"format not supported" in resp.data

def test_authenticated_upload_invalid_extension(auth_client):
    client, user_id = auth_client
    buf = io.BytesIO(b"executable payload")
    data = {'file': (buf, 'exploit.exe')}
    resp = client.post('/', data=data, content_type='multipart/form-data', follow_redirects=True)
    assert resp.status_code == 200
    assert b"Unsupported file format" in resp.data
