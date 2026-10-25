import hashlib

import pytest

from arabic_reviews.ingest import ChecksumError, verify_file


def _expected(content: bytes) -> dict:
    return {
        "size_bytes": len(content),
        "md5": hashlib.md5(content).hexdigest(),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def test_verify_file_accepts_matching_file_and_returns_fingerprint(tmp_path):
    f = tmp_path / "data.xlsx"
    f.write_bytes(b"review data")
    assert verify_file(f, _expected(b"review data")) == _expected(b"review data")


def test_verify_file_rejects_modified_file(tmp_path):
    f = tmp_path / "data.xlsx"
    f.write_bytes(b"review data, edited")
    with pytest.raises(ChecksumError):
        verify_file(f, _expected(b"review data"))
