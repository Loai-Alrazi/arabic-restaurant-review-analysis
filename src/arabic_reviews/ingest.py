"""Ingestion: fetch the source file once, verify its fingerprint, write the manifest.

The raw file is never modified; every later step reads it read-only.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import date
from pathlib import Path

from . import config


class ChecksumError(RuntimeError):
    """The local file does not match the published source fingerprint."""


def fingerprint(path: Path) -> dict:
    content = path.read_bytes()
    return {
        "size_bytes": len(content),
        "md5": hashlib.md5(content).hexdigest(),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def verify_file(path: Path, expected: dict) -> dict:
    actual = fingerprint(path)
    mismatched = [k for k in ("size_bytes", "md5", "sha256") if actual[k] != expected[k]]
    if mismatched:
        raise ChecksumError(f"{path.name}: {', '.join(mismatched)} differ from the source record")
    return actual


def ingest(raw_file: Path = config.RAW_FILE, manifest_file: Path = config.MANIFEST_FILE) -> dict:
    """Download the source file if missing, verify it and (re)write the manifest."""
    downloaded = False
    if not raw_file.exists():
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = raw_file.with_suffix(".part")
        urllib.request.urlretrieve(config.SOURCE["url"], tmp)
        verify_file(tmp, config.SOURCE)
        tmp.replace(raw_file)
        downloaded = True
    verify_file(raw_file, config.SOURCE)

    # Keep the original retrieval date so re-running ingestion is idempotent.
    retrieved = None
    if manifest_file.exists() and not downloaded:
        retrieved = json.loads(manifest_file.read_text(encoding="utf-8")).get("retrieved")
    manifest = dict(config.SOURCE, retrieved=retrieved or date.today().isoformat(), verified=True)
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest
