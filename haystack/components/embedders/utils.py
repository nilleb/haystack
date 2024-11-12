import hashlib
from typing import Any, Tuple


def sha_hash(text: str) -> str:
    """
    Hashes a string using SHA-256.

    :param text:
        Text to hash.
    :returns:
        Hashed text.
    """
    return hashlib.sha256(text.encode()).hexdigest()


class CacheProvider:
    def persist(self, key: str, value: Any):
        raise NotImplementedError

    def load(self, key: str) -> Tuple[bool, Any]:
        raise NotImplementedError
        raise NotImplementedError
