"""Utilities for small public-data downloads used by portfolio notebooks."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from time import sleep
from urllib.parse import unquote, urlparse

import requests


ByteFetcher = Callable[[str, float], bytes]


def download_bytes(url: str, timeout: float) -> bytes:
    """Download bytes with certificate handling supplied by requests/certifi."""

    parsed = urlparse(url)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).read_bytes()

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def download_bytes_with_retry(
    url: str,
    *,
    attempts: int = 3,
    timeout: float = 30,
    sleep_seconds: float = 2,
    fetcher: ByteFetcher = download_bytes,
) -> bytes:
    """Download bytes from a public URL, retrying transient failures."""

    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fetcher(url, timeout)
        except Exception as error:  # noqa: BLE001 - keep notebooks resilient to network stacks.
            last_error = error
            if attempt == attempts:
                break
            sleep(sleep_seconds)

    raise RuntimeError(f"Could not download {url!r} after {attempts} attempts") from last_error


def assert_min_bytes(content: bytes, *, min_bytes: int, label: str) -> None:
    """Fail fast when a remote payload is clearly too small to be valid."""

    actual_bytes = len(content)
    if actual_bytes < min_bytes:
        raise ValueError(
            f"{label} was unexpectedly small: {actual_bytes:,} bytes; "
            f"expected at least {min_bytes:,} bytes."
        )


def assert_sha256(content: bytes, *, expected_sha256: str, label: str) -> None:
    """Fail fast when a public source payload changes unexpectedly."""

    actual_sha256 = sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} hash changed: expected {expected_sha256}, "
            f"received {actual_sha256}."
        )
