"""
Integration tests for user authentication and session management.
"""

import unittest
try:
    from chatbot.app import app, db, User
    FLASK_AVAILABLE = True
except (ImportError, Exception):
    FLASK_AVAILABLE = False


class TestAuthIntegration(unittest.TestCase):
    def setUp(self):
        if not FLASK_AVAILABLE:
            self.skipTest("Flask or app dependencies not available in current environment.")
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()
        with app.app_context():
            db.create_all()

    def tearDown(self):
        if FLASK_AVAILABLE:
            with app.app_context():
                db.session.remove()
                db.drop_all()

    def test_login_page_renders(self):
        """Verify GET /login returns HTTP 200."""
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'NeuroScan', response.data)

    def test_unauthenticated_dashboard_redirects(self):
        """Verify unauthenticated user is redirected to /login."""
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login', response.headers.get('Location', ''))


if __name__ == '__main__':
    unittest.main()
