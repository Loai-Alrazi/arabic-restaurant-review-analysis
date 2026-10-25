"""Schema checks and the end-to-end validation run on small synthetic source frames."""

import pandas as pd
import pytest

from arabic_reviews.reports import write_outputs
from arabic_reviews.validate import SchemaError, check_integrity, check_schema, run_validation


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


def _source(rows):
    df = _raw(rows).rename(columns={"Review": "text", "Attribute": "aspect", "Result": "sentiment"})
    df.insert(0, "rid", range(len(df)))
    return df


def _tamper(result, row, col, value):
    data = result.data.copy()
    data.loc[row, col] = value
    return data


def test_integrity_check_rejects_a_record_kept_twice(result):
    data = result.data.copy()
    data.loc[1, "rid"] = data.loc[0, "rid"]
    with pytest.raises(AssertionError):
        check_integrity(data, _source(GOOD), result.logs, result.repaired_texts)


def test_integrity_check_rejects_an_unlogged_label_change(result):
    data = _tamper(result, 0, "sentiment", "سلبي")
    with pytest.raises(AssertionError):
        check_integrity(data, _source(GOOD), result.logs, result.repaired_texts)


def test_integrity_check_rejects_a_logged_change_that_is_not_the_majority(result):
    data = _tamper(result, 0, "sentiment", "سلبي")
    logs = dict(result.logs, label_resolutions=[
        {"rid": 0, "task": "sentiment", "source": "ايجابي", "resolved": "سلبي", "group_rids": [0]}
    ])
    with pytest.raises(AssertionError):
        check_integrity(data, _source(GOOD), logs, result.repaired_texts)


def test_integrity_check_rejects_a_silently_dropped_record(result):
    with pytest.raises(AssertionError):
        check_integrity(result.data.iloc[1:], _source(GOOD), result.logs, result.repaired_texts)


def test_written_outputs_are_byte_identical_across_runs(tmp_path):
    hashes = []
    for run in ("a", "b"):
        out = run_validation(_raw(GOOD), expected_rows=len(GOOD))
        d = tmp_path / run
        hashes.append(write_outputs(out, "0" * 64, d / "validated.csv", d / "lineage.json", d / "quality.md"))
    assert hashes[0] == hashes[1]
    assert b"\r\n" not in (tmp_path / "a" / "lineage.json").read_bytes()


def test_integrity_check_rejects_a_fabricated_duplicate_group(result):
    # rids 0 and 1 are different reviews; logging them as one group must be caught.
    data = result.data[result.data.rid != 1].copy()
    data.loc[data.rid == 0, "dup_group_size"] = 2
    logs = dict(result.logs)
    logs["duplicate_groups"] = logs["duplicate_groups"] + [
        {"rids": [0, 1], "aspect_votes": {}, "sentiment_votes": {}, "outcome": "kept_representative"}
    ]
    logs["excluded_rids"] = dict(logs["excluded_rids"], duplicate_copy=logs["excluded_rids"]["duplicate_copy"] + [1])
    with pytest.raises(AssertionError, match="recomputed"):
        check_integrity(data, _source(GOOD), logs, result.repaired_texts)


def test_natural_colon_prose_passes_validation_unchanged():
    out = run_validation(_raw([("رأيي باختصار: الخدمة", "الخدمة", "ايجابي")]), expected_rows=1)
    assert out.data.text.tolist() == ["رأيي باختصار: الخدمة"]
