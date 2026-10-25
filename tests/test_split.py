"""Near-duplicate grouping, grouped stratified split, leakage checks and dev-only access."""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from arabic_reviews import config, split
from arabic_reviews.text import dedup_key

LETTERS = list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")


def _random_text(rng, words=6):
    return " ".join("".join(rng.choice(LETTERS, size=rng.integers(3, 7))) for _ in range(words))


@pytest.fixture(scope="module")
def synthetic():
    """240 unrelated reviews plus near-duplicate copies (truncated, re-spelled, extended)."""
    rng = np.random.default_rng(0)
    texts = [_random_text(rng) for _ in range(240)]
    copies = []
    for i in range(0, 60, 3):
        words = texts[i].split()
        copies += [" ".join(words[1:]), texts[i].replace("ا", "أ") + "!!", texts[i] + " " + _random_text(rng, 1)]
    texts += copies
    n = len(texts)
    return pd.DataFrame({
        "rid": np.arange(n) * 7 + 3,  # rids are not positions
        "text": texts,
        "aspect": rng.choice(["a1", "a2"], size=n),
        "sentiment": rng.choice(["s1", "s2"], size=n),
    })


def test_union_find_is_transitive_and_labels_by_smallest_member():
    groups = split.union_find_groups(6, [(4, 2), (2, 5), (0, 1)])
    assert groups.tolist() == [0, 0, 2, 3, 2, 2]


def test_copy_rules_link_fragments_spacing_and_glued_copies_but_not_short_phrases():
    keys = [
        "قهوه جيده", "قهوه جيده جدا والمكان",  # short phrase inside another: not a copy
        "الاسعار مبالغ فيها كثير", "لكن الاسعار مبالغ فيها كثير جدا",  # fragment
        "مكان هادي و جميل", "مكان هادي وجميل",  # spacing only
        "الكروسان طازج والدونات لذيذه جدا اليوم", "الكروسان طازجوالدونات لذيذه جدااليوم",  # glued words
    ]
    pairs = split.copy_pairs(keys, split.similarity_matrix(keys))
    assert set(pairs) == {(2, 3), (4, 5), (6, 7)}
    assert pairs[(4, 5)] == "form"
    assert split.copy_pairs(["المكان هادئ", "مكان هادئ"], split.similarity_matrix(["المكان هادئ", "مكان هادئ"])) == {
        (0, 1): "form"}


def test_near_duplicates_link_copies_and_spelling_variants_but_not_unrelated_texts(synthetic):
    pairs = split.near_duplicate_pairs(synthetic.text.tolist())
    linked = set(zip(pairs.i, pairs.j))
    for k, i in enumerate(range(0, 60, 3)):
        for c in range(3):
            assert (i, 240 + 3 * k + c) in linked
    assert not any(j < 240 for _, j in linked)  # unrelated originals are never linked


def test_grouping_ignores_labels(synthetic):
    shuffled = synthetic.assign(aspect=synthetic.aspect.sample(frac=1, random_state=1).to_numpy())
    a = split.make_split(synthetic).assignment.group_id
    b = split.make_split(shuffled).assignment.group_id
    assert a.equals(b)


def test_synthetic_split_has_no_leakage_and_is_deterministic(synthetic):
    first, second = split.make_split(synthetic), split.make_split(synthetic.sample(frac=1, random_state=2))
    pd.testing.assert_frame_equal(first.assignment, second.assignment)
    checks = split.leakage_checks(synthetic, first.assignment, first.pairs)
    assert set(checks.values()) == {0}, checks
    assert (first.assignment.partition == "test").mean() == pytest.approx(0.2, abs=0.03)


def test_leakage_checks_detect_a_group_crossing_partitions(synthetic):
    result = split.make_split(synthetic)
    broken = result.assignment.copy()
    member = broken.index[broken.group_id.duplicated(keep=False)][0]
    broken.loc[member, "partition"] = "test" if broken.loc[member, "partition"] == "dev" else "dev"
    checks = split.leakage_checks(synthetic, broken, result.pairs)
    assert checks["partition: groups in >1 part"] >= 1
    assert checks["partition: near-duplicate pairs across parts"] >= 1


def test_leakage_checks_detect_a_shared_key_without_group_ids(synthetic):
    result = split.make_split(synthetic)
    ungrouped = result.assignment.assign(group_id=result.assignment.rid)  # pretend grouping found nothing
    ungrouped.loc[ungrouped.rid == synthetic.rid[0], "partition"] = "dev"
    ungrouped.loc[ungrouped.rid == synthetic.rid[241], "partition"] = "test"  # re-spelled copy of row 0
    checks = split.leakage_checks(synthetic, ungrouped, result.pairs.iloc[:0])
    assert checks["partition: bare keys in >1 part"] >= 1


def test_load_dev_rejects_rows_outside_the_dev_partition(tmp_path, monkeypatch):
    pd.DataFrame({"rid": [1, 2, 3], "group_id": [1, 2, 3], "partition": ["dev", "dev", "test"]}).to_csv(
        tmp_path / "split.csv", index=False)
    pd.DataFrame({"rid": [1, 3], "text": ["ا", "ب"]}).to_csv(tmp_path / "dev.csv", index=False)
    monkeypatch.setattr(config, "SPLIT_FILE", tmp_path / "split.csv")
    monkeypatch.setattr(config, "DEV_FILE", tmp_path / "dev.csv")
    with pytest.raises(RuntimeError):
        split.load_dev()
    pd.DataFrame({"rid": [1, 1], "group_id": [1, 1], "text": ["ا", "ب"]}).to_csv(tmp_path / "dev.csv", index=False)
    with pytest.raises(RuntimeError):  # duplicated rid replacing a missing one
        split.load_dev()
    pd.DataFrame({"rid": [1, 2], "group_id": [1, 9], "text": ["ا", "ب"]}).to_csv(tmp_path / "dev.csv", index=False)
    with pytest.raises(RuntimeError):  # stale group ids
        split.load_dev()


def test_only_pipeline_steps_reference_files_that_contain_test_rows():
    src = Path(split.__file__).parent
    for marker, allowed in [
        ("TEST_FILE", {"config.py", "reports.py"}),
        ("test.csv", {"config.py", "reports.py"}),
        ("VALIDATED_FILE", {"config.py", "reports.py", "__main__.py"}),
        ("validated.csv", {"config.py", "reports.py"}),
        ("PROCESSED_DIR", {"config.py"}),
    ]:
        users = {p.name for p in src.glob("*.py") if marker in p.read_text(encoding="utf-8")}
        assert users <= allowed, (marker, users - allowed)
    assert not [n for n, _ in inspect.getmembers(split, inspect.isfunction) if "test" in n.lower()]


# --- Acceptance checks on the real validated data (skipped when it is absent) ---

real = pytest.mark.skipif(
    not (config.VALIDATED_FILE.exists() and config.SPLIT_FILE.exists()), reason="validated data or split not present"
)


@pytest.fixture(scope="module")
def validated():
    return pd.read_csv(config.VALIDATED_FILE, dtype={"text": str}, keep_default_na=False)


@pytest.fixture(scope="module")
def committed():
    return pd.read_csv(config.SPLIT_FILE)


@real
def test_committed_split_is_reproduced_exactly(validated, committed):
    pd.testing.assert_frame_equal(split.make_split(validated).assignment, committed, check_dtype=False)


@real
def test_committed_split_passes_every_leakage_check(validated, committed):
    pairs = pd.read_csv(config.NEAR_DUPLICATES_FILE)
    checks = split.leakage_checks(validated, committed, pairs)
    assert set(checks.values()) == {0}, checks


def _independent_copy_signals(keys):
    """Written separately from split.copy_pairs: space-free equality and a shared run of 6+ words."""
    from difflib import SequenceMatcher

    words = [k.split() for k in keys]
    flagged = set()
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if keys[i].replace(" ", "") == keys[j].replace(" ", ""):
                flagged.add((i, j))
            elif min(len(words[i]), len(words[j])) >= 6 and set(words[i]) & set(words[j]):
                m = SequenceMatcher(None, words[i], words[j], autojunk=False)
                if m.find_longest_match(0, len(words[i]), 0, len(words[j])).size >= 6:
                    flagged.add((i, j))
    return flagged


@real
def test_no_copy_crosses_partitions_or_folds_by_an_independent_measure(validated, committed):
    """Recomputed from the texts with measures that differ from the grouping rules
    (char 2-4-gram TF-IDF >= 0.9, space-free equality, a shared run of 6+ words)."""
    df = validated.merge(committed, on="rid").sort_values("rid").reset_index(drop=True)
    keys = df.text.map(dedup_key).tolist()
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4)).fit_transform(keys)
    near = np.triu((vec @ vec.T).toarray() >= 0.9, k=1)
    flagged = _independent_copy_signals(keys) | set(zip(*map(np.ndarray.tolist, np.nonzero(near))))
    for col in ("partition", *split.CV_COLUMNS):
        crossing = [
            (int(df.rid[i]), int(df.rid[j])) for i, j in flagged
            if df[col][i] != df[col][j] and (col == "partition" or df.partition[i] == df.partition[j] == "dev")
        ]
        assert not crossing, (col, crossing[:5])


@real
def test_class_shares_match_between_dev_and_test(validated, committed):
    props = split.proportions(validated, committed)
    assert props.diff_pp.max() <= 2.0
    assert (committed.partition == "test").mean() == pytest.approx(0.2, abs=0.01)


@real
def test_load_dev_returns_exactly_the_dev_partition(committed):
    dev = split.load_dev()
    assert set(dev.rid) == set(committed.loc[committed.partition == "dev", "rid"])
    assert {"text", "aspect", "sentiment", "group_id", *split.CV_COLUMNS} <= set(dev.columns)
    assert (dev[list(split.CV_COLUMNS)] >= 0).all().all()
