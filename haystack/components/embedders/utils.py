import hashlib


def sha_hash(text: str) -> str:
    """
    Hashes a string using SHA-256.

    :param text:
        Text to hash.
    :returns:
        Hashed text.
    """
    return hashlib.sha256(text.encode()).hexdigest()
