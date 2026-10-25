"""Version 2 (docs/V2_PLAN.md): the test-data guard, the selection rule, certainty per predicted class,
the paired bootstrap over repeats and an end-to-end run on toy data."""

import json

import numpy as np
import pandas as pd
import pytest

from arabic_reviews import config, split, v2

L = v2.LABELS
POS, NEG, NEU = L


def _metrics(macro, pos=None, neg=None, neu=None):
    n = len(macro)
    return {"macro": list(macro), POS: pos or [0.8] * n, NEG: neg or [0.7] * n, NEU: neu or [0.4] * n}


def test_guard_refuses_test_files_and_records_data_files(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    for name in ("data/processed/test.csv", "reports/final_test.json", ".local/scratch/test_predictions_x.csv",
                 "data/processed/dev.csv", "reports/test_access_log.json"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x", encoding="utf-8")
    monkeypatch.setattr(config, "TEST_FILE", tmp_path / "data/processed/test.csv")
    with v2.guard() as opened:
        for name in ("data/processed/test.csv", "reports/final_test.json", ".local/scratch/test_predictions_x.csv",
                     "reports/test_access_log.json"):
            with pytest.raises(PermissionError):
                open(tmp_path / name, encoding="utf-8")
        with pytest.raises(PermissionError):
            pd.read_csv(tmp_path / "data/processed/test.csv")
        assert (tmp_path / "data/processed/dev.csv").read_text(encoding="utf-8") == "x"
    assert opened == {"data/processed/dev.csv"}
    assert (tmp_path / "data/processed/test.csv").read_text(encoding="utf-8") == "x"  # only inside the block


def test_guard_refuses_a_renamed_hard_link_to_the_test_file(tmp_path, monkeypatch):
    test = tmp_path / "test.csv"
    test.write_text("x", encoding="utf-8")
    link = tmp_path / "harmless.csv"
    try:
        link.hardlink_to(test)
    except OSError:
        pytest.skip("hard links unsupported here")
    monkeypatch.setattr(config, "TEST_FILE", test)
    with v2.guard(), pytest.raises(PermissionError):
        open(link, encoding="utf-8")


def test_eligibility_needs_gain_wins_neutral_gain_and_guardrails():
    base = _metrics([0.60] * 15)
    good = _metrics([0.62] * 15, neu=[0.43] * 15)
    assert v2.eligible(base, good)["eligible"]
    assert not v2.eligible(base, _metrics([0.62] * 15, neu=[0.41] * 15))["eligible"]  # neutral gain 0.01 < 0.02
    assert not v2.eligible(base, _metrics([0.62] * 15, neg=[0.67] * 15, neu=[0.45] * 15))["eligible"]  # negative -0.03
    assert not v2.eligible(base, _metrics([0.64] * 5 + [0.595] * 10, neu=[0.45] * 15))["eligible"]  # 5 wins
    assert v2.eligible(_metrics([0.6] * 5), _metrics([0.62] * 4 + [0.61], neu=[0.43] * 5))["eligible"]  # 5 inner folds


def test_choice_order():
    base = _metrics([0.60] * 15)
    h1 = _metrics([0.625] * 15, neu=[0.43] * 15)
    h2 = _metrics([0.62] * 15, neu=[0.43] * 15)
    h3 = _metrics([0.64] * 15, neu=[0.45] * 15)
    assert v2.choose(dict(V1=base, H1_phrases=h1, H2_cues=h2, H3_both=h3))["chosen"] == "H3_both"
    h3_small = _metrics([0.63] * 15, neu=[0.45] * 15)  # eligible vs V1 but only +0.005 over H1
    assert v2.choose(dict(V1=base, H1_phrases=h1, H2_cues=h2, H3_both=h3_small))["chosen"] == "H1_phrases"
    tie = _metrics([0.6205] * 15, neu=[0.43] * 15)  # within 0.001 of H2: the simpler H2 wins
    assert v2.choose(dict(V1=base, H1_phrases=tie, H2_cues=h2, H3_both=base))["chosen"] == "H2_cues"
    weak = _metrics([0.605] * 15)
    assert v2.choose(dict(V1=base, H1_phrases=weak, H2_cues=weak, H3_both=weak))["chosen"] == "V1"
    best_not_eligible = _metrics([0.63] * 15, neu=[0.40] * 15)  # higher gain, no neutral gain
    assert v2.choose(dict(V1=base, H1_phrases=best_not_eligible, H2_cues=h2, H3_both=base))["chosen"] == "H2_cues"


def _rows(score, correct, label, start_group):
    n = len(score)
    return pd.DataFrame(dict(score=score, pred=[label] * n, correct=correct, group=np.arange(start_group, start_group + n)))


def test_class_thresholds_need_tail_accuracy_edge_accuracy_and_group_support():
    rng = np.random.default_rng(0)
    parts = [
        _rows(rng.uniform(0.70, 0.95, 200), rng.random(200) < 0.97, POS, 0),       # high from about 0.70
        _rows(rng.uniform(0.30, 0.45, 100), rng.random(100) < 0.55, POS, 1000),    # weak low scores
        _rows(rng.uniform(0.30, 0.95, 200), rng.random(200) < 0.45, NEU, 2000),    # never reliable
        _rows(rng.uniform(0.80, 0.95, 20), np.ones(20, bool), NEG, 3000),          # too few groups
    ]
    f = pd.concat(parts, ignore_index=True)
    t = v2.class_thresholds(f.score.to_numpy(), f.pred.to_numpy(), f.correct.to_numpy(), f.group.to_numpy(), L)
    assert t[NEU] == {"high": None, "medium": None}
    assert t[NEG] == {"high": None, "medium": None}
    assert t[POS]["high"] is not None and t[POS]["high"] >= 0.45  # the weak edge vetoes the low thresholds
    assert v2.level_of(0.99, t[NEU]) == "low" and v2.level_of(0.99, t[POS]) == "high"


def test_group_support_counts_distinct_groups_not_rows():
    score = np.full(90, 0.9)
    groups = np.repeat(np.arange(10), 9)  # 90 rows, 10 groups
    t = v2.class_thresholds(score, np.array([POS] * 90), np.ones(90, bool), groups, L)
    assert t[POS]["high"] is None


def test_drop_rule_merges_a_failing_level_into_the_next_lower_one():
    thresholds = {POS: {"high": 0.6, "medium": 0.4}, NEG: {"high": 0.7, "medium": None}, NEU: {"high": None, "medium": None}}
    check = [dict(label=POS, level="high", accuracy=0.86), dict(label=NEG, level="high", accuracy=0.80),
             dict(label=POS, level="medium", accuracy=0.71)]
    used, dropped = v2.apply_drop_rule(thresholds, check)
    assert used[POS] == {"high": 0.6, "medium": 0.4}  # 0.86 is within 0.05 of 0.90
    assert used[NEG] == {"high": None, "medium": 0.7} and dropped == [f"{NEG}:high"]


def test_paired_bootstrap_resamples_groups_once_for_all_repeats():
    rng = np.random.default_rng(1)
    y = np.array(rng.choice(L, 300), dtype=object)
    groups = np.arange(300) // 2
    same = [y.copy() for _ in range(3)]
    b = v2.paired_bootstrap(y, groups, same, same, L, resamples=200)
    assert all(v["low"] == v["high"] == 0 for v in b.values())
    worse = [np.where(rng.random(300) < 0.3, POS, y) for _ in range(3)]
    b = v2.paired_bootstrap(y, groups, same, worse, L, resamples=200)
    assert b["macro"]["low"] > 0 and b["macro"]["point"] > 0


def test_calibration_of_a_perfect_and_an_underconfident_model():
    y = np.array([POS, NEG, NEU] * 10, dtype=object)
    P = np.eye(3)[[L.index(v) for v in y]]
    c = v2.calibration(P, y, L)
    assert c["ece"] == 0 and c["brier"] == 0 and c["gap"] == 0
    soft = 0.5 * P + 0.5 / 3
    assert v2.calibration(soft, y, L)["gap"] < 0  # right every time with a top probability of 0.67


def test_frozen_cue_list_is_unchanged():
    assert v2.check_frozen_cues() == v2.CUES_SHA256


POS_T = ["القهوة ممتازة", "المكان رائع وهادئ", "الخدمة سريعة ولطيفة", "الأسعار مناسبة", "الكيك لذيذ جدا"]
NEG_T = ["القهوة سيئة", "المكان مزدحم ومزعج", "الخدمة بطيئة", "الأسعار غالية", "الكيك يابس"]
NEU_T = ["القهوة ممتازة لكن غالية", "المكان جميل بس مزدحم", "الخدمة عادية", "الأسعار مقبولة نوعا ما", "الكيك لذيذ لكن يابس"]
ASP = list(config.ASPECTS)


def toy_dev():
    rows = []
    for i in range(36):
        for label, pool in ((POS, POS_T), (NEG, NEG_T), (NEU, NEU_T)):
            rows.append(dict(rid=len(rows), text=f"{pool[i % 5]} {i}", sentiment=label, aspect=ASP[i % 5]))
    dev = pd.DataFrame(rows)
    dev["group_id"] = dev.rid
    joint = dev.aspect + "|" + dev.sentiment
    for col, seed in zip(split.CV_COLUMNS, split.CV_SEEDS):
        dev[col] = split._grouped_folds(joint, dev.group_id, seed)
    return dev


def test_end_to_end_on_toy_data(tmp_path, monkeypatch):
    for name in ("LOG_FILE", "RESULTS_JSON", "RESULTS_MD", "BLIND_FILE"):
        monkeypatch.setattr(v2, name, tmp_path / getattr(v2, name).name)
    monkeypatch.setattr(v2, "V2_DIR", tmp_path)
    r = v2.run_experiments(toy_dev(), check_v1=False)
    assert r["selection"]["chosen"] in ("V1", "H1_phrases", "H2_cues", "H3_both")
    assert len(r["nested"]["paired"]["class_change"]) == 3 and sum(r["nested"]["chosen_counts"].values()) == 15
    assert r["released"] == (r["selection"]["chosen"] if r["adopted"] else "V1")
    assert not any("test" in p.split("/")[-1] for p in r["inputs"])
    saved = json.loads(v2.RESULTS_JSON.read_text(encoding="utf-8"))
    assert saved["certainty"]["sentiment"]["thresholds"].keys() == set(L)
    assert "# Version 2" in v2.RESULTS_MD.read_text(encoding="utf-8")
    log = v2.read_log()
    assert len(log) == 11 and log.exp_id.is_unique  # 4 candidates x 2 schemes, procedure, aspect x 2
    again = v2.run_experiments(toy_dev(), check_v1=False)  # logged rows are checked, not appended
    assert len(v2.read_log()) == 11 and again["selection"]["chosen"] == r["selection"]["chosen"]


def test_released_rule_never_widens_a_version_1_level():
    rng = np.random.default_rng(2)
    f = pd.concat([_rows(rng.uniform(0.40, 0.95, 400), rng.random(400) < 0.97, POS, 0),
                   _rows(rng.uniform(0.60, 0.95, 300), rng.random(300) < 0.80, NEG, 1000)], ignore_index=True)
    args = (f.score.to_numpy(), f.pred.to_numpy(), f.correct.to_numpy(), f.group.to_numpy(), L)
    free = v2.class_thresholds(*args)
    kept = v2.class_thresholds(*args, floor={"high": 0.60, "medium": 0.50})
    assert free[POS]["high"] < 0.60 and kept[POS]["high"] == 0.60  # registered rule widens, released rule does not
    assert kept[NEG]["high"] is None or kept[NEG]["high"] > 0.60  # 80% right: high is raised or removed
    for label in L:
        for level in ("high", "medium"):
            assert kept[label][level] is None or kept[label][level] >= {"high": 0.60, "medium": 0.50}[level]
