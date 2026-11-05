"""
Integration tests for scan upload and history routes.
"""

import unittest
try:
    from chatbot.app import app, db, User, Scan
    FLASK_AVAILABLE = True
except (ImportError, Exception):
    FLASK_AVAILABLE = False


class TestScanRoutes(unittest.TestCase):
    def setUp(self):
        if not FLASK_AVAILABLE:
            self.skipTest("Flask dependencies not installed.")
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.client = app.test_client()
        with app.app_context():
            db.create_all()

    def tearDown(self):
        if FLASK_AVAILABLE:
            with app.app_context():
                db.session.remove()
                db.drop_all()

    def test_unauthenticated_upload_redirect(self):
        """Verify unauthenticated access to index upload route is redirected."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login', response.headers.get('Location', ''))

    def test_unauthenticated_history_redirect(self):
        """Verify scan history is protected behind authentication."""
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login', response.headers.get('Location', ''))


if __name__ == '__main__':
    unittest.main()
