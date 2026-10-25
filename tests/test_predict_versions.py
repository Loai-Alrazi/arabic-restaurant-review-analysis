"""Versioned inference on the tiny pipelines of test_predict_fixture: a version-2 card with thresholds per
predicted class (a level may not exist for a class) next to the unchanged version-1 card."""

import json

import pytest

from arabic_reviews import predict as predict_module
from arabic_reviews.predict import certainty_level, predict, thresholds_for
from test_predict_fixture import tiny_models  # noqa: F401  (fixture)


def _write_v2_card(root, sentiment_thresholds):
    card = json.loads((root / "models" / "model_card.json").read_text(encoding="utf-8"))
    for task in ("sentiment", "aspect"):
        card["tasks"][task]["file"] = f"models/v2/{task}.joblib"
        labels = card["tasks"][task]["labels"]
        by_class = sentiment_thresholds if task == "sentiment" else {l: {"high": 0.5, "medium": 0.3} for l in labels}
        card["tasks"][task]["certainty"] = dict(thresholds_by_class=by_class)
    (root / "models" / "v2").mkdir()
    for task in ("sentiment", "aspect"):
        (root / "models" / "v2" / f"{task}.joblib").write_bytes((root / "models" / f"{task}.joblib").read_bytes())
    (root / "models" / "model_card_v2.json").write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")
    predict_module.model_card.cache_clear()
    predict_module.load_models.cache_clear()


@pytest.fixture
def with_v2(tiny_models, monkeypatch):  # noqa: F811
    monkeypatch.setattr(predict_module, "CARD_FILE_V2", tiny_models / "models" / "model_card_v2.json")
    return tiny_models


def test_a_level_without_threshold_never_appears(with_v2):
    _write_v2_card(with_v2, {"ايجابي": {"high": None, "medium": None}, "سلبي": {"high": 0.0, "medium": 0.0}})
    r = predict("القهوة ممتازة والطعم رائع", version="v2")
    assert r.version == "v2" and r.sentiment.label == "ايجابي"
    assert r.sentiment.level == "low" and r.sentiment.second_label is not None
    bad = predict("القهوة سيئة والطعم مر", version="v2")
    assert bad.sentiment.label == "سلبي" and bad.sentiment.level == "high"


def test_version_1_is_the_default_and_keeps_its_thresholds(with_v2):
    _write_v2_card(with_v2, {"ايجابي": {"high": None, "medium": None}, "سلبي": {"high": None, "medium": None}})
    r = predict("القهوة ممتازة والطعم رائع")
    assert r.version == "v1"
    card = predict_module.model_card()
    assert r.sentiment.level == certainty_level(r.sentiment.score, card["tasks"]["sentiment"]["certainty"]["thresholds"])


def test_available_versions_need_card_and_models(with_v2):
    assert predict_module.available_versions() == ["v1"]
    _write_v2_card(with_v2, {"ايجابي": {"high": 0.9, "medium": 0.5}, "سلبي": {"high": 0.9, "medium": 0.5}})
    assert predict_module.available_versions() == ["v1", "v2"]
    (with_v2 / "models" / "v2" / "aspect.joblib").unlink()
    assert predict_module.available_versions() == ["v1"]
    with pytest.raises(FileNotFoundError, match="v2-build"):
        predict("القهوة ممتازة", version="v2")


def test_unknown_version_is_refused(with_v2):
    with pytest.raises(ValueError):
        predict("القهوة ممتازة", version="v3")


def test_thresholds_for_reads_either_card_layout():
    assert thresholds_for({"certainty": {"thresholds": {"high": 0.6, "medium": 0.5}}}, "x") == {"high": 0.6, "medium": 0.5}
    by_class = {"certainty": {"thresholds_by_class": {"x": {"high": None, "medium": 0.4}}}}
    assert thresholds_for(by_class, "x") == {"high": None, "medium": 0.4}
    assert certainty_level(0.99, {"high": None, "medium": 0.4}) == "medium"
    assert certainty_level(0.99, {"high": None, "medium": None}) == "low"
