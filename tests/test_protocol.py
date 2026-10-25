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


def test_fingerprint_covers_configuration_thresholds_data_and_plan():
    p = dict(dev_sha256="d", split_sha256="s", labels={"sentiment": ["a", "b"]}, level_rule={"min_bin": 30},
             bootstrap={"resamples": 2000}, fresh_seeds=[100], versions={"scikit-learn": "1"},
             tasks={"sentiment": dict(final={"name": "a", "params": {"C": 0.3}}, reference={"name": "r"},
                                      baselines=[{"name": "d"}], pipelines={"final": {"clf__max_iter": 2000}},
                                      certainty=dict(thresholds={"high": 0.6, "medium": 0.45}),
                                      expected_range={"final": {"mean": 0.6}})},
             clean_rerun=[{"rerun_sha": "x"}])
    base = protocol.fingerprint(p)
    edits = [("tasks", "sentiment", "certainty", "thresholds", "high"), ("tasks", "sentiment", "final", "params", "C"),
             ("tasks", "sentiment", "pipelines", "final", "clf__max_iter"), ("dev_sha256",), ("split_sha256",),
             ("bootstrap", "resamples"), ("versions", "scikit-learn"), ("level_rule", "min_bin")]
    for path in edits:
        changed = json.loads(json.dumps(p))
        node = changed
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = "changed"
        assert protocol.fingerprint(changed) != base, path
    measured = json.loads(json.dumps(p))
    measured["tasks"]["sentiment"]["expected_range"]["final"]["mean"] = 0.7
    measured["clean_rerun"] = []
    assert protocol.fingerprint(measured) == base  # measurements are reported, not frozen inputs


def test_sparse_band_cannot_start_a_level_and_edges_match_binning():
    table = pd.DataFrame(dict(lo=[0.5, 0.55, 0.6], n=[100, 29, 100], accuracy=[0.5, 0.0, 0.95]))
    assert protocol.threshold(table, 0.9) == 0.6
    th = {"high": 0.6, "medium": 0.5}
    score = 0.6 - 1e-11
    assert protocol.band_table(np.array([score]), np.array([1])).set_index("lo").loc[0.6, "n"] == 1
    assert protocol.certainty_level(score, th) == "high"


def test_complexity_gate_on_the_regularisation_grid():
    grid = [0.1, 0.3, 1, 3, 10]
    flat = [[0.5] * 15 for _ in grid]
    assert experiments.gate_complex_pick(grid, flat, 1) == (1, None)  # stronger than default: no check
    small = [s[:] for s in flat]
    small[3] = [0.505] * 15
    pick, gate = experiments.gate_complex_pick(grid, small, 3)
    assert pick == 2 and not gate["accepted"]  # weaker, gain below 0.01: default kept
    big = [s[:] for s in flat]
    big[3] = [0.52] * 15
    assert experiments.gate_complex_pick(grid, big, 3)[0] == 3


def test_error_sample_categories_and_gate():
    oof = pd.DataFrame(dict(rid=range(10), y=list("aaaaabbbbb"), pred=list("abababbbab")))
    sample = experiments.oof_error_sample(oof, 3, seed=7)
    assert len(sample) == 3 and (sample.y != sample.pred).all()
    assert sample.rid.tolist() == experiments.oof_error_sample(oof, 3, seed=7).rid.tolist()
    cats = pd.DataFrame(dict(task=["sentiment"] * 4 + ["aspect"] * 3, rid=range(7), model=["m"] * 4 + ["k"] * 3,
                             category=["lexical_variety"] * 2 + ["negation", "other"]
                             + ["lexical_variety", "multi_aspect", "multi_aspect"]))
    gate = experiments.gate_decision(cats)
    assert gate["sentiment"]["opens"] and gate["sentiment"]["share"] == 0.5
    assert not gate["aspect"]["opens"] and not gate["aspect"]["largest"]
    md = experiments.categories_markdown(cats, {"sentiment": "m", "aspect": "k2"})
    assert "gate opens" in md and "gate stays closed" in md and "`k2`" in md


def test_git_sha_flags_edited_log_and_untracked_code(tmp_path, monkeypatch):
    import subprocess

    def git(*a):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=tmp_path, check=True,
                       capture_output=True)

    log = tmp_path / "reports" / "experiments.csv"
    log.parent.mkdir()
    log.write_text("a\n1\n", encoding="utf-8")
    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    monkeypatch.setattr(experiments.config, "ROOT", tmp_path)
    monkeypatch.setattr(experiments, "EXPERIMENTS_FILE", log)
    assert not experiments._git_sha().endswith("+dirty")
    log.write_text("a\n1\n2\n", encoding="utf-8")  # appended rows are fine
    assert not experiments._git_sha().endswith("+dirty")
    log.write_text("a\n9\n2\n", encoding="utf-8")  # an edited row is not
    assert experiments._git_sha().endswith("+dirty")
    log.write_text("a\n1\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "new.py").write_text("x = 1\n", encoding="utf-8")
    assert experiments._git_sha().endswith("+dirty")


def test_build_protocol_end_to_end_on_a_toy_log(tmp_path, monkeypatch):
    """Whole M3 -> M4 -> protocol chain on toy data with its own log; the test partition path
    points to a missing file, so any read of it would fail."""
    monkeypatch.setattr(experiments, "EXPERIMENTS_FILE", tmp_path / "experiments.csv")
    monkeypatch.setattr(experiments, "_git_sha", lambda: "abc1234")
    monkeypatch.setattr(protocol.config, "TEST_FILE", tmp_path / "missing" / "test.csv")
    (tmp_path / "dev.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "split.csv").write_text("partition\ndev\ntest\n", encoding="utf-8")
    monkeypatch.setattr(protocol.config, "DEV_FILE", tmp_path / "dev.csv")
    monkeypatch.setattr(protocol.config, "SPLIT_FILE", tmp_path / "split.csv")
    dev = grouped_toy_dev()
    experiments.run_pending(dev, experiments.REGISTRY)
    for task in experiments.TASKS:
        experiments.resolve_axes(task, dev)
        r = experiments.resolve_m4(task, dev)
        assert [f["candidate"] for f in r["families"]] == ["m4_linsvm", "m4_cnb"]
        assert len(r["tuning"]) == 5 and len(r["tfidf_steps"]) == 2
        assert r["picked"] == r["one_se"] or (r["gate"] is not None and not r["gate"]["accepted"])
    p = protocol.build_protocol(dev, rerun=True)
    assert len(p["clean_rerun"]) == 2 * 5 and all(x["max_fold_diff"] == 0 for x in p["clean_rerun"])
    assert p["test_rows"] == 1 and p["fingerprint"] == protocol.fingerprint(p)
    for task, t in p["tasks"].items():
        assert t["reference"]["name"] == protocol.REFERENCE and t["expected_range"]["final"]["n"] == 15
        assert set(t["certainty"]["thresholds"]) == {"high", "medium"}
        assert "clf__max_iter" in t["pipelines"]["final"]
    again = protocol.build_protocol(dev, rerun=True)  # reruns are reused, after their checks
    assert again["fingerprint"] == p["fingerprint"]
    md = protocol.protocol_markdown(p)
    assert p["fingerprint"] in md and "One-shot test procedure" in md


@pytest.mark.skipif(not split.config.DEV_FILE.exists(), reason="needs the local development partition")
def test_committed_error_categories_match_the_seeded_sample():
    cats = pd.read_csv(experiments.ERROR_CATEGORIES_FILE, keep_default_na=False)
    dev = split.load_dev()
    for task, g in cats.groupby("task"):
        base = experiments.resolve_m4(task)["base"]
        name = g.model.iloc[0]
        value = float(name.split("_C")[1])
        spec = dict(base, name=name, params={"C": value})
        oof = experiments.oof_predictions(spec, dev, task)
        sample = experiments.oof_error_sample(oof, experiments.ERROR_SAMPLE_SIZE[task])
        assert sorted(sample.rid) == sorted(g.rid)
        merged = g.merge(sample, on="rid")
        assert (merged.gold == merged.y).all() and (merged.pred_x == merged.pred_y).all()


