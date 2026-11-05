"""
Global Pytest Fixtures and Lifecycle Management for NeuroScan AI.
Ensures an isolated, consistent database schema and application context for all tests.
"""
import pytest
from chatbot.app import app, db


@pytest.fixture(autouse=True)
def ensure_database_schema():
    """
    Autouse fixture that runs before every test in the suite.
    Ensures database tables exist even if a preceding test called db.drop_all().
    """
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.app_context():
        db.create_all()
        yield
        try:
            db.session.rollback()
        except Exception:
            pass
