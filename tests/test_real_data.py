"""Acceptance checks on the real source file (skipped when it has not been ingested)."""

import re

import pandas as pd
import pytest

from arabic_reviews import config
from arabic_reviews.validate import LABEL_SCAN_RE, LABEL_SUFFIX_RE, run_validation

pytestmark = pytest.mark.skipif(not config.RAW_FILE.exists(), reason="source file not ingested")


@pytest.fixture(scope="module")
def raw():
    return pd.read_excel(config.RAW_FILE, sheet_name=0, dtype=object, engine="openpyxl")


def test_validation_is_deterministic_and_reconciles(raw):
    first, second = run_validation(raw), run_validation(raw)
    pd.testing.assert_frame_equal(first.data, second.data)
    assert first.lineage == second.lineage
    assert first.lineage["raw_rows"] == len(first.data) + sum(first.lineage["excluded_by_reason"].values())


def test_no_label_strings_remain_in_validated_text(raw):
    texts = run_validation(raw).data.text
    assert not texts.map(lambda t: bool(LABEL_SUFFIX_RE.search(t))).any()
    assert not texts.map(lambda t: bool(LABEL_SCAN_RE.search(t))).any()


# Written independently of the production patterns: any label word right after a
# delimiter, with any spelling variant, anywhere in the text.
INDEPENDENT_SCAN = re.compile(
    r"[;:؛]\s*(?:ال(?:م[أا]كولات|مشروبات|خدم[ةه]|جلسات|[اأ]سعار)|[اأإ]يجاب[يى]|سلب[يى]|محايد)"
)


def test_independent_scan_finds_no_label_string_after_repair(raw):
    data = run_validation(raw).data
    hits = data[data.text.map(lambda t: bool(INDEPENDENT_SCAN.search(t)))]
    assert hits.empty, hits.rid.tolist()
