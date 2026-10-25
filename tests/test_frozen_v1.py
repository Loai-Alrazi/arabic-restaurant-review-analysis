"""Version 1 stays frozen: its decision log, protocol, one-shot test results and model card are
byte-identical to the merged release, and the saved version-1 pipelines still give the recorded
predictions (golden values recorded before version 2 changed the inference module).

This is a version-1 integrity check: it hashes the bytes of the published files and parses none of
them; no version-2 code or decision reads the test results."""

import hashlib
import json
from pathlib import Path

import pytest

from arabic_reviews import config

FROZEN = {
    "reports/experiments.csv": "4686991fbde28be4693f6b66b9a8b7e9968ca1ec51de9e2092db1707c5fd4a0b",
    "reports/protocol.json": "cfda3fcb93e15f3f59a766541bc1fa28088fd51832b682ec5642cb07b7920107",
    "reports/final_protocol.md": "7a7e6e0df27719e0338b2872a27b2ed239727c256496f160c4ccff48d1ed9a68",
    "reports/final_test.json": "30a6372e9f096bde452354a4f865b6f8b090a2b26aba4acb6de081193d8987a0",
    "reports/final_test.md": "768e499335ba072555c8702ff2cc0bc56e8010c610c04a29acc547a64c8a7138",
    "reports/test_access_log.json": "f4c56ca7c5c658fa98c47866ffa7e7103fd7f4d9da3dcedfdcbf768950ab38bc",
    "reports/model_digests.json": "fa8792486e43a5c2dfedd615707711426693b6a26adff8942d5a7747cf89050f",
    "models/model_card.json": "a791365971e4464c56d54b174ea1bb8bc9aa59a089f35b8d3411c8dd2cc8a65d",
    "reports/error_categories.csv": "e5774c0d60e2e1152fcabc8a1b4ff1c13c91d93a9d7663cec736d441bc20ef4f",
    "reports/models.md": "679babf44956d24614793727774c362e55f42b859b8753893c4764bbcb834254",
    "reports/ablations.md": "4d648277fdb1d0e743749ef4a76111fe68613bb44629209fe8fdb67a864d9d4d",
    "reports/error_analysis.md": "b60dd500e9d0af450554a27f72cb641cf3810d27160e02f4191cee5cd5f11077",
}
GOLDEN = Path(__file__).parent / "data" / "v1_golden_predictions.json"
needs_models = pytest.mark.skipif(
    not (config.ROOT / "models" / "sentiment.joblib").exists(), reason="saved models not built (final --check)")


@pytest.mark.parametrize("path", sorted(FROZEN))
def test_frozen_version_1_file_is_unchanged(path):
    data = (config.ROOT / path).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == FROZEN[path]


@needs_models
def test_version_1_predictions_match_the_golden_values():
    from arabic_reviews.predict import predict

    for row in json.loads(GOLDEN.read_text(encoding="utf-8")):
        r = predict(row["text"])
        assert r.notes == row["notes"]
        for task in ("sentiment", "aspect"):
            got, want = getattr(r, task), row[task]
            assert (got.label, got.level, got.second_label) == (want["label"], want["level"], want["second_label"])
            assert abs(got.score - want["score"]) < 1e-6
