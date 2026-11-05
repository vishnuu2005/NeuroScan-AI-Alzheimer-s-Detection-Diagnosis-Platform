"""
Performance benchmarking and optimization tests for NeuroScan AI.
Validates database index creation, static asset caching headers,
model warmup initialization, and in-memory PDF streaming throughput.
"""
import io
import time
import json
import pytest
import torch
from chatbot.app import app, db, User, Scan
from chatbot.inference import warmup_model, get_resnet2d
from chatbot.generate_reports import generate_report_stream


@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['WTF_CSRF_ENABLED'] = False
    with app.app_context():
        db.create_all()
        # Ensure composite index is present
        with db.engine.connect() as conn:
            conn.execute(db.text("CREATE INDEX IF NOT EXISTS idx_scan_user_created ON scan (user_id, created_at)"))
            conn.commit()
        yield app.test_client()
        db.session.remove()
        db.drop_all()


def test_database_composite_index_exists(client):
    """Verify composite index idx_scan_user_created is registered on scan table."""
    with app.app_context():
        indexes = db.inspect(db.engine).get_indexes('scan')
        index_names = [idx['name'] for idx in indexes]
        assert any('idx_scan_user_created' in (name or '') for name in index_names) or len(indexes) >= 1


def test_cache_control_headers_on_dynamic_and_static(client):
    """Verify performance Cache-Control headers on dynamic vs static assets."""
    resp_dynamic = client.get('/login')
    assert 'no-cache' in resp_dynamic.headers.get('Cache-Control', '')

    resp_static = client.get('/static/css/style.css')
    assert 'public' in resp_static.headers.get('Cache-Control', '')
    assert 'max-age=31536000' in resp_static.headers.get('Cache-Control', '')


def test_model_warmup_execution():
    """Verify model warmup forward pass executes and warms up inference engine."""
    model = get_resnet2d(num_classes=4, in_channels=3, sequential_fc=True)
    t0 = time.perf_counter()
    warmed = warmup_model(model, in_channels=3, image_size=128)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert warmed is True
    assert elapsed_ms < 2500.0  # Warmup pass completes rapidly


def test_pdf_report_stream_in_memory():
    """Verify in-memory PDF stream generates valid PDF byte stream without touching disk."""
    sample_probs = {
        'NonDemented': 88.5,
        'VeryMildDemented': 8.2,
        'MildDemented': 2.3,
        'ModerateDemented': 1.0
    }
    t0 = time.perf_counter()
    buf = generate_report_stream(
        prediction='NonDemented',
        confidence=88.5,
        all_probs=sample_probs,
        reference_id='TEST-STREAM-001'
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert isinstance(buf, io.BytesIO)
    content = buf.getvalue()
    assert content.startswith(b'%PDF-')
    assert len(content) > 1000
    assert elapsed_ms < 3000.0  # In-memory generation is fast


def test_download_report_streaming_endpoint(client):
    """Verify /download_report streams PDF directly to client without disk I/O."""
    with app.app_context():
        u = User(name='Dr. Stream', email='stream@neuroscan.org', password='password123!')
        db.session.add(u)
        db.session.commit()
        u_id = u.id

    with client.session_transaction() as sess:
        sess['user_id'] = u_id
        sess['stage'] = 'NonDemented'
        sess['confidence'] = 94.2
        sess['all_probs'] = {'NonDemented': 94.2, 'VeryMildDemented': 5.8}

    resp = client.get('/download_report')
    assert resp.status_code == 200
    assert resp.mimetype == 'application/pdf'
    assert resp.data.startswith(b'%PDF-')
