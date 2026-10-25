import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arabic_reviews import config, experiments, features, split

POS = ["الخدمة ممتازة وسريعة", "القهوة لذيذة جدا", "مكان جميل وهادئ", "الاسعار مناسبة وممتازة"]
NEG = ["الخدمة سيئة وبطيئة", "القهوة باردة وسيئة", "مكان مزدحم ومزعج", "الاسعار غالية جدا"]


def toy_dev(n_per=30):
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n_per):
        for label, pool in (("ايجابي", POS), ("سلبي", NEG)):
            rows.append(dict(text=f"{pool[i % 4]} {i}", sentiment=label,
                             aspect=["الخدمة", "المشروبات", "الجلسات", "الاسعار"][i % 4]))
    dev = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
    for k, col in enumerate(split.CV_COLUMNS):
        dev[col] = rng.permutation(np.arange(len(dev)) % 5)
    return dev


def test_spec_validation():
    with pytest.raises(ValueError):
        features.build_pipeline(dict(model="logreg", representation="bogus"))
    with pytest.raises(ValueError):
        features.build_pipeline(dict(model="svm"))


def test_normalizer_is_stateless_and_keeps_negation():
    out = features.Normalizer("full").fit_transform(["لا يعجبني!!"])
    assert "لا" in out[0].split()


def test_vocabulary_is_learned_inside_the_fold():
    pipe = features.build_pipeline(dict(model="logreg", representation="word11"))
    pipe.fit(["كلمة اولى", "كلمة ثانية"], ["a", "b"])
    vocab = pipe.named_steps["features"].vocabulary_
    assert "ثالثة" not in vocab
    pipe.predict(["ثالثة"])
    assert "ثالثة" not in pipe.named_steps["features"].vocabulary_


def test_cv_scores_use_all_15_folds_and_are_reproducible():
    dev = toy_dev()
    spec = experiments.REGISTRY[2]
    a = experiments.cv_scores(spec, dev, "sentiment")
    b = experiments.cv_scores(spec, dev, "sentiment")
    assert len(a["f1"]) == 15 and a["f1"] == b["f1"]


def test_dummy_baselines_are_below_the_model():
    dev = toy_dev()
    by_name = {s["name"]: s for s in experiments.REGISTRY}
    dummy = np.mean(experiments.cv_scores(by_name["dummy_most_frequent"], dev, "sentiment")["f1"])
    model = np.mean(experiments.cv_scores(by_name["logreg_word11"], dev, "sentiment")["f1"])
    assert model > dummy + 0.2


def test_acceptance_rule_needs_gain_and_wins():
    base = [0.5] * 15
    assert experiments.compare(base, [0.52] * 15)["accepted"]
    assert not experiments.compare(base, [0.505] * 15)["accepted"]  # too small
    skewed = [0.5] * 15
    cand = [0.7] * 5 + [0.49] * 10  # mean gain 0.03 but only 5 wins
    assert experiments.compare(skewed, cand)["mean_delta"] > 0.01
    assert not experiments.compare(skewed, cand)["accepted"]


def test_run_pending_is_append_only(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "EXPERIMENTS_FILE", tmp_path / "experiments.csv")
    dev = toy_dev()
    registry = experiments.REGISTRY[:3]
    first = experiments.run_pending(dev, registry)
    assert len(first) == 2 * 3
    assert list(pd.read_csv(tmp_path / "experiments.csv").columns) == experiments.COLUMNS
    assert experiments.run_pending(dev, registry).empty
    more = experiments.run_pending(dev, experiments.REGISTRY[:4])
    assert list(more.exp_id) == ["sentiment/logreg_word12", "aspect/logreg_word12"]
    log = experiments.read_log()
    assert len(json.loads(log.fold_scores.iloc[0])) == 15


def test_decide_chain_walks_winners():
    def row(name, scores):
        return dict(exp_id=f"sentiment/{name}", fold_scores=json.dumps(scores))

    log = pd.DataFrame([row("a", [0.5] * 15), row("b", [0.4] * 15), row("c", [0.6] * 15), row("d", [0.605] * 15)])
    steps = experiments.decide_chain(log, "sentiment", ["a", "b", "c", "d"])
    assert [s["accepted"] for s in steps] == [False, True, False]
    assert steps[-1]["final_winner"] == "c"


def test_experiment_code_cannot_reach_the_held_out_test():
    for name in ("experiments.py", "features.py"):
        src = (Path(config.ROOT) / "src" / "arabic_reviews" / name).read_text(encoding="utf-8")
        assert not re.search(r"TEST_FILE|test\.csv|VALIDATED_FILE|validated\.csv|SPLIT_FILE|\[.partition", src), name


def test_validation_rows_never_reach_fit():
    from sklearn.base import BaseEstimator, ClassifierMixin

    seen = []

    class Recorder(ClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            seen.append(set(X))
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            seen[-1].isdisjoint(set(X)) or pytest.fail("validation text seen in fit")
            return np.array([self.classes_[0]] * len(X))

    dev = toy_dev()
    dev["text"] = [f"t{i}" for i in range(len(dev))]
    orig = experiments.build_pipeline
    experiments.build_pipeline = lambda spec: Recorder()
    try:
        experiments.cv_scores(experiments.REGISTRY[0], dev, "sentiment")
    finally:
        experiments.build_pipeline = orig
    assert len(seen) == 15 and all(len(s) < len(dev) for s in seen)


def test_axis_markdown_lists_decisions():
    log = pd.DataFrame([dict(exp_id=f"{t}/{n}", fold_scores=json.dumps([v] * 15))
                        for t in experiments.TASKS for n, v in (("a", 0.5), ("b", 0.6))])
    md = experiments.axis_markdown(log, "Axis X", ["a", "b"])
    assert "| b | a | +0.1000 | 15/15 | accepted |" in md and "Winner for aspect: `b`" in md
