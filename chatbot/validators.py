import os
from PIL import Image

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'tif', 'tiff'}

MAGIC_NUMBERS = {
    'jpeg': [b'\xff\xd8\xff'],
    'png': [b'\x89PNG\r\n\x1a\n'],
    'webp': [b'RIFF'],
    'tiff': [b'II*\x00', b'MM\x00*']
}

def allowed_file(filename):
    """Check if file extension is among permitted types."""
    if not filename or '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def verify_magic_bytes(stream_or_path):
    """
    Examines file header magic bytes to verify legitimate image signatures.
    Prevents executable or script masquerading as image files.
    """
    header = b''
    if isinstance(stream_or_path, str):
        with open(stream_or_path, 'rb') as f:
            header = f.read(16)
    else:
        pos = stream_or_path.tell()
        stream_or_path.seek(0)
        header = stream_or_path.read(16)
        stream_or_path.seek(pos)

    if not header or len(header) < 4:
        return False, "File is empty or corrupted"

    # JPEG signature
    if header.startswith(b'\xff\xd8\xff'):
        return True, "jpeg"
    
    # PNG signature
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        return True, "png"

    # WEBP signature: 'RIFF....WEBP'
    if header.startswith(b'RIFF') and len(header) >= 12 and header[8:12] == b'WEBP':
        return True, "webp"

    # TIFF signatures (little-endian and big-endian)
    if header.startswith(b'II*\x00') or header.startswith(b'MM\x00*'):
        return True, "tiff"

    return False, "Invalid image file header signature"

def validate_mri_image(file_storage_or_path, min_dim=64, max_dim=4096):
    """
    Comprehensive validation for uploaded MRI scan:
    - Checks file extension against whitelist
    - Validates binary magic bytes
    - Verifies structural integrity via Pillow
    - Enforces realistic medical image dimension boundaries
    """
    if isinstance(file_storage_or_path, str):
        filename = os.path.basename(file_storage_or_path)
        path = file_storage_or_path
        stream = None
    else:
        filename = file_storage_or_path.filename
        path = None
        stream = file_storage_or_path.stream

    if not allowed_file(filename):
        return False, f"File format not supported. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS)).upper()}"

    valid_magic, magic_info = verify_magic_bytes(path or stream)
    if not valid_magic:
        return False, f"Security check failed: {magic_info}"

    try:
        if path:
            with Image.open(path) as img:
                img.verify()
            # Reopen to read dimensions after verify
            with Image.open(path) as img:
                width, height = img.size
                format_name = img.format
        else:
            stream.seek(0)
            with Image.open(stream) as img:
                img.verify()
            stream.seek(0)
            with Image.open(stream) as img:
                width, height = img.size
                format_name = img.format
            stream.seek(0)

        if width < min_dim or height < min_dim:
            return False, f"Image resolution ({width}x{height}) is too small for clinical inference. Minimum required is {min_dim}x{min_dim}."

        if width > max_dim or height > max_dim:
            return False, f"Image resolution ({width}x{height}) exceeds maximum supported bounds ({max_dim}x{max_dim})."

        return True, {
            "format": format_name,
            "dimensions": (width, height),
            "width": width,
            "height": height
        }
    except Exception as e:
        return False, f"Corrupted or invalid image file: {str(e)}"

