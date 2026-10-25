import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from arabic_reviews import config, experiments, final, protocol, split
from test_experiments import toy_dev
from test_protocol import grouped_toy_dev


def test_per_class_f1_matches_sklearn_with_fixed_labels_and_zero_division():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y, p = rng.integers(0, 4, 40), rng.integers(0, 4, 40)
        y[y == 3] = 0  # class 3 never true: an absent class must score 0, as in the CV
        expected = f1_score(y, p, labels=[0, 1, 2, 3], average=None, zero_division=0)
        assert np.allclose(final.per_class_f1(y, p, 4), expected)
    assert np.allclose(final.per_class_f1(np.array([0, 0]), np.array([0, 0]), 3), [1, 0, 0])


def test_resamples_draw_whole_groups_as_many_as_there_are():
    groups = np.array([5, 5, 7, 9, 9, 9, 2])
    idx = final.resample_indices(groups, 200, seed=0)
    assert len(idx) == 200
    for i in idx[:50]:
        drawn = groups[i]
        # every drawn group appears with all of its rows, and the draw holds 4 groups
        counts = pd.Series(drawn).value_counts()
        sizes = pd.Series(groups).value_counts()
        assert all(counts[g] % sizes[g] == 0 for g in counts.index)
        assert sum(counts[g] // sizes[g] for g in counts.index) == 4
    again = final.resample_indices(groups, 200, seed=0)
    assert all(np.array_equal(a, b) for a, b in zip(idx, again))


def test_bootstrap_pairs_models_on_the_same_resamples():
    labels = ["a", "b", "c"]
    rng = np.random.default_rng(1)
    y = rng.choice(labels, 300)
    groups = np.arange(300) // 3
    noisy = np.where(rng.uniform(size=300) < 0.8, y, rng.choice(labels, 300))
    preds = {"final": noisy, "reference": noisy.copy(), "dummy_most_frequent": np.full(300, "a"),
             "dummy_stratified": rng.choice(labels, 300)}
    b = final.bootstrap(y, preds, labels, groups, dict(resamples=300, seed=0, level=0.95))
    assert b["paired"]["reference"]["low"] == b["paired"]["reference"]["high"] == 0
    assert b["paired"]["reference"]["share_le_0"] == 1.0
    assert b["paired"]["dummy_most_frequent"]["low"] > 0
    lo, hi = b["macro_f1"]["final"]["low"], b["macro_f1"]["final"]["high"]
    point = f1_score(y, noisy, labels=labels, average="macro", zero_division=0)
    assert lo < point < hi and set(b["per_class_f1"]["final"]) == set(labels)
    d = final.decide(b["paired"])
    assert d["reference"] == "not distinguishable" and all(d["beats_dummies"].values()) and d["success"]


def test_decision_wording_follows_the_frozen_rule():
    def paired(ref, dummy):
        return {"reference": dict(zip(("low", "high"), ref)),
                **{d: dict(zip(("low", "high"), dummy)) for d in final.DUMMIES}}
    assert final.decide(paired((0.01, 0.05), (0.1, 0.2)))["reference"] == "above"
    below = final.decide(paired((-0.05, -0.01), (0.1, 0.2)))
    assert below["reference"] == "below" and not below["success"]
    tie = final.decide(paired((-0.01, 0.02), (0.0, 0.2)))  # a lower bound of exactly 0 does not beat
    assert tie["reference"] == "not distinguishable" and not tie["success"]
    assert final.position(0.5, dict(low=0.6, high=0.7)) == "below"
    assert final.position(0.6, dict(low=0.6, high=0.7)) == "inside"
    assert final.position(0.71, dict(low=0.6, high=0.7)) == "above"


def test_a_second_run_needs_a_written_reason():
    final.check_access([], None)
    with pytest.raises(final.RepeatedTestRun):
        final.check_access([{"run": 1}], None)
    with pytest.raises(final.RepeatedTestRun):
        final.check_access([{"run": 1}], "   ")
    final.check_access([{"run": 1}], "the test file was rebuilt after a documented split bug")


def test_fingerprint_mismatch_stops_the_run():
    with pytest.raises(final.ProtocolMismatch):
        final.check_fingerprint({"fingerprint": "a"}, {"fingerprint": "b"})
    final.check_fingerprint({"fingerprint": "a"}, {"fingerprint": "a"})


def test_the_test_partition_is_opened_in_one_place_only():
    src = Path(final.__file__).read_text(encoding="utf-8")
    assert src.count("config.TEST_FILE") == 1
    reader = re.search(r"def (\w+)\([^)]*\)[^:]*:(?:(?!\ndef ).)*config\.TEST_FILE", src, re.S).group(1)
    assert reader == "read_test_once"
    for name in ("protocol.py", "experiments.py", "features.py", "eda.py"):
        other = (Path(final.__file__).parent / name).read_text(encoding="utf-8")
        assert "TEST_FILE" not in other and "import final" not in other and "from .final" not in other, name


def toy_test():
    test = toy_dev(n_per=8).drop(columns=list(split.CV_COLUMNS))
    test["rid"] = 1000 + np.arange(len(test))
    test["group_id"] = 1000 + np.arange(len(test)) // 2
    return test[["rid", "text", "aspect", "sentiment", "group_id"]]


@pytest.fixture
def toy_world(tmp_path, monkeypatch):
    """A toy log, a frozen toy protocol and toy dev/test partitions; every real path is redirected."""
    monkeypatch.setattr(experiments, "EXPERIMENTS_FILE", tmp_path / "experiments.csv")
    monkeypatch.setattr(experiments, "_git_sha", lambda: "abc1234")
    monkeypatch.setattr(final, "provenance", lambda: "abc1234")
    dev, test = grouped_toy_dev(), toy_test()
    (tmp_path / "dev.csv").write_text("x\n", encoding="utf-8")
    assignment = pd.concat([dev[["rid", "group_id"]].assign(partition="dev"),
                            test[["rid", "group_id"]].assign(partition="test")])
    assignment.to_csv(tmp_path / "split.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False, encoding="utf-8")
    for name, path in [("DEV_FILE", "dev.csv"), ("SPLIT_FILE", "split.csv"), ("TEST_FILE", "test.csv")]:
        monkeypatch.setattr(config, name, tmp_path / path)
    for mod, name, path in [(protocol, "PROTOCOL_JSON", "protocol.json"), (protocol, "PROTOCOL_MD", "protocol.md"),
                            (final, "ACCESS_LOG", "access.json"), (final, "FINAL_JSON", "final_test.json"),
                            (final, "FINAL_MD", "final_test.md"), (final, "MODELS_DIR", "models"),
                            (final, "SCRATCH_DIR", "scratch")]:
        monkeypatch.setattr(mod, name, tmp_path / path)
    monkeypatch.setattr(split, "load_dev", lambda: dev.copy())
    monkeypatch.setitem(protocol.BOOTSTRAP, "resamples", 200)
    experiments.run_pending(dev, experiments.REGISTRY)
    for task in experiments.TASKS:
        experiments.resolve_axes(task, dev)
        experiments.resolve_m4(task, dev)
    protocol.write_protocol(protocol.build_protocol(dev, rerun=False))
    return tmp_path, dev, test


def test_final_run_end_to_end_on_toy_data(toy_world, monkeypatch):
    tmp, dev, test = toy_world
    checked = final.run_final(check_only=True)  # verifies and fits, but never opens the test file
    assert checked["test_read"] is False and not (tmp / "access.json").exists()

    r = final.run_final()
    log = json.loads((tmp / "access.json").read_text(encoding="utf-8"))
    assert len(log) == 1 and log[0]["run"] == 1 and log[0]["test_rows"] == len(test)
    assert log[0]["fingerprint"] == json.loads((tmp / "protocol.json").read_text(encoding="utf-8"))["fingerprint"]
    for task in experiments.TASKS:
        t = r["tasks"][task]
        assert set(t["metrics"]) == set(final.ROLES)
        assert t["position"]["final"] in ("below", "inside", "above")
        assert abs(sum(lv["share"] for lv in t["levels"]) - 1) < 1e-9
        assert len(t["metrics"]["final"]["confusion"]) == len(experiments.LABELS[task])
        model = joblib.load(tmp / "models" / f"{task}.joblib")
        y = test[experiments.TASKS[task]].to_numpy()
        assert np.mean(model.predict(test.text.to_numpy()) == y) == pytest.approx(t["metrics"]["final"]["accuracy"])
    assert 0 <= r["joint"]["accuracy"] <= 1
    card = json.loads((tmp / "models" / "model_card.json").read_text(encoding="utf-8"))
    assert card["fingerprint"] == log[0]["fingerprint"] and set(card["tasks"]) == set(experiments.TASKS)
    md = (tmp / "final_test.md").read_text(encoding="utf-8")
    assert log[0]["test_sha256"] in md and "Macro-F1" in md
    assert (tmp / "final_test.json").exists()

    # a repeated run is refused before anything is read; the first result stays in place
    first = (tmp / "final_test.json").read_bytes()
    monkeypatch.setattr(config, "TEST_FILE", tmp / "missing.csv")
    with pytest.raises(final.RepeatedTestRun):
        final.run_final()
    assert len(json.loads((tmp / "access.json").read_text(encoding="utf-8"))) == 1
    assert (tmp / "final_test.json").read_bytes() == first


def test_changed_protocol_stops_before_the_test_is_read(toy_world, monkeypatch):
    tmp, _, _ = toy_world
    stored = json.loads((tmp / "protocol.json").read_text(encoding="utf-8"))
    stored["fingerprint"] = "0" * 64
    (tmp / "protocol.json").write_text(json.dumps(stored), encoding="utf-8")
    monkeypatch.setattr(config, "TEST_FILE", tmp / "missing.csv")
    with pytest.raises(final.ProtocolMismatch):
        final.run_final()
    assert not (tmp / "access.json").exists()


def test_test_rows_must_match_the_published_split(toy_world):
    tmp, _, test = toy_world
    test.assign(group_id=test.group_id + 1).to_csv(tmp / "test.csv", index=False, encoding="utf-8")
    with pytest.raises(RuntimeError, match="split"):
        final.run_final()
