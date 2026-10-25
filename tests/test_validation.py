"""Schema checks and the end-to-end validation run on small synthetic source frames."""

import pandas as pd
import pytest

from arabic_reviews.validate import SchemaError, check_schema, run_validation


def _raw(rows):
    return pd.DataFrame(rows, columns=["Review", "Attribute", "Result"])


GOOD = [
    ("القهوة لذيذة جدا;المشروبات;ايجابي", "المشروبات", "ايجابي"),
    ("الموظف تأخر علينا", "الخدمة", "سلبي"),
    ("Loved it", "الجلسات", "ايجابي"),
    ("الاسعار مرتفعه", "الاسعار", "سلبي"),
    ("الاسعار مرتفعة!!", "الاسعار", "سلبي"),
    ("الاسعار مرتفعة", "الاسعار", "محايد"),
    ("المكان رايق", "الجلسات", "ايجابي"),
    ("المكان رايق", "الجلسات", "محايد"),
]


def test_schema_accepts_expected_columns_and_labels():
    check_schema(_raw(GOOD), expected_rows=len(GOOD))


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"Review": ["نص"], "Attribute": ["الخدمة"]}),
        _raw([("نص", "الديكور", "ايجابي")]),
        _raw([("نص", "الخدمة", "ممتاز")]),
        _raw([(None, "الخدمة", "ايجابي")]),
        _raw([(123, "الخدمة", "ايجابي")]),
    ],
)
def test_schema_rejects_missing_columns_unknown_labels_nulls_and_non_text(frame):
    with pytest.raises(SchemaError):
        check_schema(frame, expected_rows=len(frame))


def test_schema_rejects_unexpected_row_count():
    with pytest.raises(SchemaError):
        check_schema(_raw(GOOD), expected_rows=len(GOOD) + 1)


@pytest.fixture
def result():
    return run_validation(_raw(GOOD), expected_rows=len(GOOD))


def test_embedded_label_is_removed_from_kept_text(result):
    assert result.data.set_index("rid").loc[0, "text"] == "القهوة لذيذة جدا"


def test_text_without_arabic_is_excluded(result):
    assert 2 not in set(result.data.rid)


def test_duplicate_group_keeps_one_representative_with_majority_labels(result):
    kept = result.data.set_index("rid")
    assert 3 in kept.index and not {4, 5} & set(kept.index)
    assert kept.loc[3, "sentiment"] == "سلبي" and kept.loc[3, "dup_group_size"] == 3


def test_duplicate_group_with_tied_labels_is_excluded_whole(result):
    assert not {6, 7} & set(result.data.rid)


def test_counts_reconcile_raw_equals_kept_plus_excluded(result):
    excluded = sum(result.lineage["excluded_by_reason"].values())
    assert result.lineage["raw_rows"] == len(GOOD) == len(result.data) + excluded
    assert result.lineage["excluded_by_reason"] == {
        "no_arabic_text": 1,
        "duplicate_copy": 2,
        "duplicate_group_label_tie": 2,
    }


def test_every_kept_label_matches_the_source_label(result):
    raw = _raw(GOOD)
    for row in result.data.itertuples():
        assert (row.aspect, row.sentiment) == (raw.Attribute[row.rid], raw.Result[row.rid])


def test_embedded_label_disagreeing_with_official_label_is_logged_not_used():
    raw = _raw([("الطلب تأخر;الخدمة;ايجابي", "الخدمة", "سلبي")])
    out = run_validation(raw, expected_rows=1)
    assert out.data.sentiment.tolist() == ["سلبي"]
    assert out.logs["embedded_label_conflicts"] == [
        {"rid": 0, "task": "sentiment", "embedded": "ايجابي", "official": "سلبي"}
    ]


def test_majority_resolution_that_changes_a_label_is_logged():
    raw = _raw(
        [
            ("المكان رايق", "الجلسات", "محايد"),
            ("المكان رايق", "الجلسات", "ايجابي"),
            ("المكان رايق", "الجلسات", "ايجابي"),
        ]
    )
    out = run_validation(raw, expected_rows=3)
    assert out.data[["rid", "sentiment"]].values.tolist() == [[0, "ايجابي"]]
    assert out.logs["label_resolutions"] == [
        {"rid": 0, "task": "sentiment", "source": "محايد", "resolved": "ايجابي", "group_rids": [0, 1, 2]}
    ]
