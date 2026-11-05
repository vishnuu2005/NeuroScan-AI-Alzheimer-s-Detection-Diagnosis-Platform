"""
Validation helpers for authentication, user input, and file uploads.
"""

import re
from typing import Tuple, Optional

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def validate_email(email: str) -> Tuple[bool, Optional[str]]:
    """Validate email format and length."""
    if not email or not isinstance(email, str):
        return False, "Email address is required."
    email = email.strip()
    if len(email) > 120:
        return False, "Email address cannot exceed 120 characters."
    if not EMAIL_REGEX.match(email):
        return False, "Please enter a valid email address (e.g., user@example.com)."
    return True, None


def validate_password(password: str) -> Tuple[bool, Optional[str]]:
    """Validate password length and basic complexity."""
    if not password or not isinstance(password, str):
        return False, "Password is required."
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if len(password) > 128:
        return False, "Password cannot exceed 128 characters."
    if not any(char.isdigit() for char in password) and not any(not char.isalnum() for char in password):
        return False, "Password should contain at least one number or special character."
    return True, None


def validate_registration(name: str, email: str, password: str) -> Tuple[bool, Optional[str]]:
    """Validate complete registration payload."""
    if not name or not isinstance(name, str) or not name.strip():
        return False, "Full name is required."
    if len(name.strip()) < 2:
        return False, "Full name must be at least 2 characters long."
    if len(name.strip()) > 100:
        return False, "Full name cannot exceed 100 characters."

    valid_email, email_err = validate_email(email)
    if not valid_email:
        return False, email_err

    valid_pwd, pwd_err = validate_password(password)
    if not valid_pwd:
        return False, pwd_err

    return True, None
