"""Fernet symmetric encryption for storing sensitive credentials.

The encryption key is loaded from the ENCRYPTION_KEY environment variable.
If no key is configured, a temporary key is generated (data will be lost on restart).
"""

from __future__ import annotations

import logging

from cryptography.fernet import Fernet, InvalidToken

from config import get_settings

logger = logging.getLogger(__name__)

_fernet: Fernet | None = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is not None:
        return _fernet

    settings = get_settings()
    key = settings.encryption_key

    if not key:
        logger.warning(
            "ENCRYPTION_KEY not set — generating a temporary key. "
            "Encrypted data will be unrecoverable after restart."
        )
        key = Fernet.generate_key().decode()

    try:
        _fernet = Fernet(key.encode() if isinstance(key, str) else key)
    except Exception:
        logger.error("Invalid ENCRYPTION_KEY — generating temporary key as fallback.")
        _fernet = Fernet(Fernet.generate_key())

    return _fernet


def encrypt(plaintext: str) -> str:
    """Encrypt a string and return a URL-safe base64-encoded token."""
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt(token: str) -> str:
    """Decrypt a Fernet token back to the original string.

    Raises ValueError if the token is invalid or tampered with.
    """
    try:
        return _get_fernet().decrypt(token.encode()).decode()
    except InvalidToken as e:
        raise ValueError("Failed to decrypt — invalid token or wrong key") from e
