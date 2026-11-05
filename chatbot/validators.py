"""
Validation helpers for authentication, user input, and file uploads.
"""

import os
from typing import Tuple, Optional

ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
MAX_UPLOAD_BYTES = 16 * 1024 * 1024  # 16 MB


def validate_image_extension(filename: str) -> Tuple[bool, Optional[str]]:
    """Verify that uploaded file has an approved image extension."""
    if not filename or '.' not in filename:
        return False, "Uploaded file must have a valid extension (e.g. .jpg, .png)."
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
        return False, f"Unsupported file format '{ext}'. Allowed formats: {allowed}."
    return True, None
