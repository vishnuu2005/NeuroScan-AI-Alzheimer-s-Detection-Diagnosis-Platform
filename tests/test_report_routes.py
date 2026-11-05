"""
Integration tests for PDF report downloading and authorization.
"""

import unittest
try:
    from chatbot.app import app, db, User, Scan
    FLASK_AVAILABLE = True
except (ImportError, Exception):
    FLASK_AVAILABLE = False


class TestReportRoutes(unittest.TestCase):
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

    def test_unauthenticated_download_redirect(self):
        """Verify unauthenticated user cannot access report download."""
        response = self.client.get('/download_report/1')
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login', response.headers.get('Location', ''))

    def test_unauthenticated_session_report_redirect(self):
        """Verify unauthenticated user cannot access session report."""
        response = self.client.get('/download_report')
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login', response.headers.get('Location', ''))


if __name__ == '__main__':
    unittest.main()
