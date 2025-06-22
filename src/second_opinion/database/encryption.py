"""
Encryption manager for secure field-level encryption in conversation storage.
"""

import base64
import os

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..config.settings import get_settings


class EncryptionManager:
    """Manages field-level encryption for conversation data."""

    def __init__(self):
        self._encryption_keys: dict[str, Fernet] = {}
        self._current_key_id: str | None = None
        self._initialize_encryption()

    def _initialize_encryption(self) -> None:
        """Initialize encryption with master key from environment or generate new."""
        settings = get_settings()

        # Try to get master key from environment
        master_key = os.environ.get("SECOND_OPINION_MASTER_KEY")

        if not master_key:
            # Generate new master key for development
            master_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
            # In development, store in settings for consistency
            if hasattr(settings, "development_mode") and settings.development_mode:
                os.environ["SECOND_OPINION_MASTER_KEY"] = master_key

        # Create primary encryption key
        self._current_key_id = "primary_v1"
        key = self._derive_key(master_key, self._current_key_id)
        self._encryption_keys[self._current_key_id] = Fernet(key)

    def _derive_key(self, master_key: str, key_id: str) -> bytes:
        """Derive encryption key from master key and key ID."""
        # Use key_id as salt for key derivation
        salt = key_id.encode("utf-8").ljust(16, b"0")[:16]

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )

        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        return key

    def encrypt(self, data: str) -> tuple[str, str]:
        """
        Encrypt data and return (encrypted_data, key_id).

        Returns:
            Tuple of (base64_encrypted_data, key_id_used)
        """
        if (
            not self._current_key_id
            or self._current_key_id not in self._encryption_keys
        ):
            raise ValueError("No encryption key available")

        fernet = self._encryption_keys[self._current_key_id]
        encrypted_bytes = fernet.encrypt(data.encode("utf-8"))
        encrypted_b64 = base64.b64encode(encrypted_bytes).decode("utf-8")

        return encrypted_b64, self._current_key_id

    def decrypt(self, encrypted_data: str, key_id: str) -> str:
        """
        Decrypt data using the specified key ID.

        Args:
            encrypted_data: Base64 encoded encrypted data
            key_id: Key ID that was used for encryption

        Returns:
            Decrypted string data
        """
        if key_id not in self._encryption_keys:
            # Try to recreate the key if we have the master key
            master_key = os.environ.get("SECOND_OPINION_MASTER_KEY")
            if master_key:
                key = self._derive_key(master_key, key_id)
                self._encryption_keys[key_id] = Fernet(key)
            else:
                raise ValueError(f"Encryption key {key_id} not available")

        fernet = self._encryption_keys[key_id]
        encrypted_bytes = base64.b64decode(encrypted_data.encode("utf-8"))
        decrypted_bytes = fernet.decrypt(encrypted_bytes)

        return decrypted_bytes.decode("utf-8")

    def get_current_key_id(self) -> str:
        """Get the current key ID for new encryptions."""
        if not self._current_key_id:
            raise ValueError("No current encryption key set")
        return self._current_key_id

    def rotate_key(self, new_key_id: str) -> None:
        """
        Create a new encryption key for future encryptions.

        Note: This doesn't re-encrypt existing data - that would require
        a separate migration process.
        """
        master_key = os.environ.get("SECOND_OPINION_MASTER_KEY")
        if not master_key:
            raise ValueError("Master key not available for key rotation")

        key = self._derive_key(master_key, new_key_id)
        self._encryption_keys[new_key_id] = Fernet(key)
        self._current_key_id = new_key_id


# Global encryption manager instance
_encryption_manager: EncryptionManager | None = None


def get_encryption_manager() -> EncryptionManager:
    """Get the global encryption manager instance."""
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    # Type: ignore since we just ensured it's not None
    return _encryption_manager  # type: ignore[return-value]
