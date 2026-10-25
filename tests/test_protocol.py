import json

import numpy as np
import pandas as pd
import pytest

from arabic_reviews import experiments, protocol, split
from test_experiments import toy_dev


def grouped_toy_dev():
    dev = toy_dev().assign(rid=lambda d: np.arange(len(d)))
    dev["group_id"] = dev.rid // 2  # pairs of rows that must stay together
    return dev


def test_threshold_is_the_first_edge_from_which_every_populated_band_reaches_the_target():
    rng = np.random.default_rng(0)
    score = rng.uniform(0.3, 1.0, 20000)
    correct = rng.uniform(size=score.size) < score  # accuracy rises with the score
    th = protocol.certainty_thresholds(score, correct)
    assert th == {"high": 0.9, "medium": 0.7}
    assert protocol.certainty_level(0.95, th) == "high"
    assert protocol.certainty_level(0.9, th) == "high"
    assert protocol.certainty_level(0.75, th) == "medium"
    assert protocol.certainty_level(0.69, th) == "low"


def test_a_populated_weak_band_vetoes_a_threshold_and_a_sparse_one_does_not():
    table = pd.DataFrame(dict(lo=[0.5, 0.55, 0.6, 0.65], n=[100, 100, 100, 5], accuracy=[0.95, 0.80, 0.95, 0.0]))
    assert protocol.threshold(table, 0.9) == 0.6
    assert protocol.threshold(table.assign(n=[100, 100, 100, 100]), 0.9) == 1.0


def test_band_table_puts_a_score_of_one_in_the_last_band():
    table = protocol.band_table(np.array([1.0, 0.951, 0.05, 0.0499]), np.array([1, 1, 0, 1]))
    assert table.n.sum() == 4
    assert table.set_index("lo").loc[0.95, "n"] == 2 and table.set_index("lo").loc[0.0, "n"] == 1


def test_level_summary_shares_add_up():
    th = {"high": 0.8, "medium": 0.5}
    rows = protocol.level_summary(np.array([0.9, 0.85, 0.6, 0.3]), np.array([1, 0, 1, 0]), th)
    assert [r["n"] for r in rows] == [2, 1, 1]
    assert rows[0]["accuracy"] == 0.5 and abs(sum(r["share"] for r in rows) - 1) < 1e-12


def test_fresh_folds_keep_groups_whole_and_leave_selection_folds_untouched():
    dev = grouped_toy_dev()
    fresh = protocol.with_fresh_folds(dev)
    for col in protocol.FRESH_COLUMNS:
        assert sorted(set(fresh[col])) == [0, 1, 2, 3, 4]
        assert (fresh.groupby("group_id")[col].nunique() == 1).all()
    assert fresh[list(split.CV_COLUMNS)].equals(dev[list(split.CV_COLUMNS)])
    assert not fresh[protocol.FRESH_COLUMNS[0]].equals(fresh[protocol.FRESH_COLUMNS[1]])


def test_fold_metrics_and_joint_accuracy():
    s = pd.DataFrame(dict(rid=[1, 2, 3, 4], fold=[0, 0, 1, 1], y=list("aabb"), pred=list("abbb")))
    a = pd.DataFrame(dict(rid=[1, 2, 3, 4], fold=[0, 0, 1, 1], y=list("xxyy"), pred=list("xxyx")))
    assert protocol.fold_metrics([s], ["a", "b"]) == pytest.approx([1 / 3, 0.5])  # absent labels score 0, as in cv_scores
    assert protocol.joint_fold_accuracy([s], [a]) == [0.5, 0.5]
    r = protocol.expected_range([0.5, 0.7])
    assert r["mean"] == pytest.approx(0.6) and r["low"] == pytest.approx(0.6 - 2 * r["sd"]) and r["n"] == 2


def test_clean_rerun_refuses_a_dirty_tree(monkeypatch):
    monkeypatch.setattr(experiments, "_git_sha", lambda: "abc1234+dirty")
    with pytest.raises(RuntimeError):
        protocol.clean_rerun(grouped_toy_dev(), {})


def test_fingerprint_covers_configuration_and_thresholds():
    tasks = {"sentiment": dict(final={"name": "a", "params": {"C": 0.3}}, reference={"name": "r"},
                               certainty=dict(thresholds={"high": 0.6, "medium": 0.45}))}
    base = protocol.fingerprint(tasks)
    changed = json.loads(json.dumps(tasks))
    changed["sentiment"]["certainty"]["thresholds"]["high"] = 0.65
    assert protocol.fingerprint(changed) != base
    changed = json.loads(json.dumps(tasks))
    changed["sentiment"]["final"]["params"]["C"] = 1
    assert protocol.fingerprint(changed) != base


def test_build_protocol_end_to_end_on_toy_data(tmp_path, monkeypatch):
    for name in ("DEV_FILE", "SPLIT_FILE"):
        path = tmp_path / f"{name}.csv"
        path.write_text("x\n", encoding="utf-8")
        monkeypatch.setattr(protocol.config, name, path)
    p = protocol.build_protocol(grouped_toy_dev(), rerun=False)
    for task, t in p["tasks"].items():
        assert t["final"]["model"] == "logreg" and t["reference"]["name"] == protocol.REFERENCE
        assert t["expected_range"]["final"]["n"] == 15
        assert set(t["certainty"]["thresholds"]) == {"high", "medium"}
    md = protocol.protocol_markdown(p)
    assert p["fingerprint"] in md and "One-shot test procedure" in md
