import unittest
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch
from urllib.error import URLError

from ml_portfolio.data_sources import (
    assert_sha256,
    assert_min_bytes,
    download_bytes,
    download_bytes_with_retry,
)


class DataSourceHelperTests(unittest.TestCase):
    def test_assert_min_bytes_accepts_expected_payload_size(self):
        assert_min_bytes(b"abcde", min_bytes=5, label="toy payload")

    def test_assert_min_bytes_rejects_short_payloads(self):
        with self.assertRaisesRegex(ValueError, "toy payload"):
            assert_min_bytes(b"abcd", min_bytes=5, label="toy payload")

    def test_assert_sha256_rejects_changed_payload(self):
        with self.assertRaisesRegex(ValueError, "toy payload hash changed"):
            assert_sha256(
                b"changed",
                expected_sha256="0" * 64,
                label="toy payload",
            )

    def test_download_bytes_with_retry_uses_later_attempt_after_transient_failure(self):
        calls = []

        def fake_fetcher(url: str, timeout: float) -> bytes:
            calls.append((url, timeout))
            if len(calls) == 1:
                raise URLError("temporary outage")
            return b"payload"

        result = download_bytes_with_retry(
            "https://example.test/data.zip",
            attempts=2,
            timeout=7,
            sleep_seconds=0,
            fetcher=fake_fetcher,
        )

        self.assertEqual(result, b"payload")
        self.assertEqual(
            calls,
            [("https://example.test/data.zip", 7), ("https://example.test/data.zip", 7)],
        )

    def test_download_bytes_uses_requests_with_timeout_and_status_check(self):
        response = Mock()
        response.content = b"payload"

        with patch("ml_portfolio.data_sources.requests.get", return_value=response) as get:
            result = download_bytes("https://example.test/data.zip", timeout=7)

        self.assertEqual(result, b"payload")
        get.assert_called_once_with("https://example.test/data.zip", timeout=7)
        response.raise_for_status.assert_called_once_with()

    def test_download_bytes_supports_file_urls_for_network_free_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.csv"
            path.write_bytes(b"payload")

            result = download_bytes(path.as_uri(), timeout=7)

        self.assertEqual(result, b"payload")


if __name__ == "__main__":
    unittest.main()
