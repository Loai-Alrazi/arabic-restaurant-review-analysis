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


def test_negation_marking_and_stopwords_keep_negations():
    from arabic_reviews import text

    assert text.mark_negation("الخدمة ما كانت جيدة") == "الخدمة ما NEG_كانت جيدة"
    assert text.mark_negation("لا !") == "لا !"  # a symbol is not marked
    assert text.mark_negation("") == ""
    assert text.remove_stopwords("القهوة في المكان ما كانت جيدة") == "القهوة المكان ما جيدة"
    assert not text.STOPWORDS & text.NEGATIONS


def test_text_stats_are_counts_of_the_raw_text():
    from arabic_reviews import text

    stats = text.text_stats("ما أحلاه!!! ؟ 😀😀")
    assert stats[2] == 3.0 and stats[3] == 1.0 and stats[4] == 2.0 and stats[5] == 1.0


@pytest.mark.parametrize("change", [
    dict(normalisation="full"), dict(symbols="separate"), dict(symbols="remove"),
    dict(digits="token"), dict(negation=True), dict(stopwords=True), dict(textstats=True),
    dict(class_weight="balanced"),
])
def test_every_axis_option_builds_a_working_pipeline(change):
    spec = experiments._spec("x", "logreg", "word11+char25", **{k: v for k, v in change.items()
                                                              if k != "class_weight"},
                             class_weight=change.get("class_weight"))
    pipe = features.build_pipeline(spec)
    X = ["الخدمة ممتازة 😀 !", "ما عجبني المكان 5", "القهوة لذيذة جدا", "الاسعار غالية ؟"]
    pipe.fit(X, ["ايجابي", "سلبي", "ايجابي", "سلبي"])
    assert list(pipe.predict(["ما عجبني"])) in (["ايجابي"], ["سلبي"])


def test_symbol_tokens_reach_the_word_vocabulary_only_when_separated():
    X = ["جميل😀", "سيء!"]
    plain = features.build_pipeline(experiments._spec("p", "logreg", "word11"))
    sep = features.build_pipeline(experiments._spec("s", "logreg", "word11", symbols="separate"))
    for pipe in (plain, sep):
        pipe.fit(X, ["a", "b"])
    assert "😀" not in plain.named_steps["features"].vocabulary_
    assert "😀" in sep.named_steps["features"].vocabulary_


def test_resolve_axes_logs_each_candidate_once_and_is_reproducible(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "EXPERIMENTS_FILE", tmp_path / "experiments.csv")
    dev = toy_dev(12)
    experiments.run_pending(dev, experiments.REGISTRY)
    n_candidates = sum(len(v) for _, _, v in experiments.AXES)
    rows, final = experiments.resolve_axes("sentiment", dev)
    assert len(rows) == n_candidates and len(experiments.read_log()) == 2 * 6 + n_candidates
    again, final_again = experiments.resolve_axes("sentiment")  # read-only from the log
    assert [r["accepted"] for r in rows] == [r["accepted"] for r in again]
    assert final["name"] == final_again["name"]
    assert "Axes 2-6" in experiments.axes_markdown({"sentiment": (rows, final)})


def test_negation_and_stopwords_handle_diacritics_double_negation_and_whitespace():
    from arabic_reviews import text

    assert text.remove_stopwords("كَلام جميل بَارد وَاضح") == "كَلام جميل بَارد وَاضح"
    assert text.mark_negation("مَا عجبني") == "مَا NEG_عجبني"
    assert text.mark_negation("لا ما عجبني") == "لا ما NEG_عجبني"
    assert text.mark_negation("ما\nعجبني") == "ما\nNEG_عجبني"
    assert text.mark_negation("ما. جيد") == "ما. جيد"  # a sentence boundary ends the scope
    assert text.text_stats("لَا")[5] == 1.0


def test_text_stats_use_raw_text_even_when_symbols_are_removed():
    spec = experiments._spec("x", "logreg", "word11", symbols="remove", textstats=True)
    pipe = features.build_pipeline(spec)
    pipe.fit(["جميل !!", "سيء 😀", "جيد", "رديء"], ["a", "b", "a", "b"])
    stats = pipe.named_steps["features"].transformer_list[1][1].named_steps["stats"]
    assert stats.transform(["جميل !!"])[0][2] == 2.0


def test_symbols_remove_and_digits_token_output():
    from arabic_reviews import text

    assert text.normalize("جميل😀!", symbols="remove") == "جميل"
    assert text.normalize("سعر ١٢٠ ريال", digits="token") == "سعر <NUM> ريال"


def test_resolve_axes_follows_accepted_winner_and_rejects_stale_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "EXPERIMENTS_FILE", tmp_path / "experiments.csv")
    dev = toy_dev(12)
    experiments.run_pending(dev, experiments.REGISTRY)
    experiments.resolve_axes("sentiment", dev)
    log = experiments.read_log()
    # make a2_full a clear winner: later axes must then be compared with it
    idx = log.index[log.exp_id == "sentiment/a2_full"][0]
    log.at[idx, "fold_scores"] = json.dumps([0.99] * 15)
    log.to_csv(tmp_path / "experiments.csv", index=False)
    # a2_full is now accepted, so a3 is built on it; the logged a3 rows came from the old
    # chain and are reported as stale instead of being silently reused
    with pytest.raises(ValueError, match="a3_symbols_separate"):
        experiments.resolve_axes("sentiment")


# ---- M4 ----

@pytest.mark.parametrize("model,params", [("linsvm", {"C": 0.3}), ("cnb", {"alpha": 0.3}), ("logreg", {"C": 3})])
def test_model_families_build_and_learn(model, params):
    dev = toy_dev()
    spec = dict(name="t", model=model, representation="word11", normalisation="none",
                class_weight=None if model == "cnb" else "balanced", params=params, notes="")
    res = experiments.cv_scores(spec, dev, "sentiment")
    assert len(res["f1"]) == 15 and np.mean(res["f1"]) > 0.65  # toy data has no neutral class: the cap is 2/3


def test_cnb_rejects_class_weight_and_tfidf_options_reach_the_vectorizer():
    with pytest.raises(ValueError):
        features.build_pipeline(dict(model="cnb", class_weight="balanced"))
    pipe = features.build_pipeline(dict(model="linsvm", representation="word11+char25",
                                        tfidf={"min_df": 2, "sublinear_tf": True}))
    for _, vec in pipe.named_steps["features"].transformer_list:
        assert vec.min_df == 2 and vec.sublinear_tf


def test_tfidf_option_is_part_of_the_logged_identity():
    plain = experiments._spec("a", "linsvm")
    tuned = dict(plain, tfidf={"min_df": 2})
    assert experiments.row_identity(plain)["params"] != experiments.row_identity(tuned)["params"]
    assert "tfidf" not in experiments.row_identity(plain)["params"]  # old log rows stay valid


def test_one_standard_error_rule_prefers_the_simplest_close_candidate():
    means, stds = [0.80, 0.84, 0.85, 0.845, 0.83], [0.02] * 5  # SE = 0.02 / sqrt(15) ~ 0.0052
    assert experiments.one_se_choice(means, stds) == 2          # only the best is within the bar
    assert experiments.one_se_choice([0.846, 0.848, 0.85], [0.03] * 3) == 0  # SE 0.0077: simplest wins


def test_oof_predictions_cover_each_row_once_and_use_other_folds():
    dev = toy_dev().assign(rid=lambda d: np.arange(len(d)))
    spec = experiments.REGISTRY[2]
    oof = experiments.oof_predictions(spec, dev, "sentiment")
    assert sorted(oof.rid) == list(range(len(dev))) and oof.pred.notna().all()
    table = experiments.per_class_table(oof, experiments.LABELS["sentiment"])
    assert table.support.sum() == len(dev)
    assert "| class |" in experiments.error_markdown("sentiment", spec, oof)
