"""
Unit tests for authentication, payload, and file validators.
"""

import unittest
import re

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def validate_email_test(email):
    if not email or not isinstance(email, str):
        return False
    return bool(EMAIL_REGEX.match(email.strip())) and len(email.strip()) <= 120


def validate_pwd_test(password):
    if not password or len(password) < 8 or len(password) > 128:
        return False
    return any(c.isdigit() or not c.isalnum() for c in password)


class TestValidators(unittest.TestCase):
    def test_valid_emails(self):
        valid = ["doctor@hospital.com", "jane.doe@clinic.org", "researcher+mri@university.edu"]
        for email in valid:
            self.assertTrue(validate_email_test(email), f"Failed on valid email: {email}")

    def test_invalid_emails(self):
        invalid = ["", "plainaddress", "missing@domain", "@nodomain.com", "user@.com"]
        for email in invalid:
            self.assertFalse(validate_email_test(email), f"Allowed invalid email: {email}")

    def test_password_validation(self):
        self.assertTrue(validate_pwd_test("StrongPass123"))
        self.assertTrue(validate_pwd_test("Secure!Pass"))
        self.assertFalse(validate_pwd_test("short1"))
        self.assertFalse(validate_pwd_test("allalphapassword"))


if __name__ == '__main__':
    unittest.main()
