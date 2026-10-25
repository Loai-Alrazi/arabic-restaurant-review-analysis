"""Inference, refusals and the demo page on a tiny pipeline built in the test, so they run in CI too.

The real saved pipelines are covered by tests/test_predict.py when they exist locally; here a handful
of synthetic reviews gives two small pipelines with the same structure and a minimal model card.
"""

import json
import platform
import sys
from pathlib import Path

import joblib
import numpy
import pytest

from arabic_reviews import config, features, predict as predict_module
from arabic_reviews.predict import InvalidReview, check_review, predict

SPEC = dict(model="logreg", params=dict(C=10), normalisation="none", representation="word11+char25",
            class_weight="balanced")
TEXTS = [
    ("القهوة ممتازة والطعم رائع", "ايجابي", "القهوة"),
    ("القهوة لذيذة جدا", "ايجابي", "القهوة"),
    ("القهوة سيئة والطعم مر", "سلبي", "القهوة"),
    ("القهوة باردة ومرة", "سلبي", "القهوة"),
    ("الخدمة ممتازة والموظف لطيف", "ايجابي", "الخدمة"),
    ("الخدمة سريعة ومحترمة", "ايجابي", "الخدمة"),
    ("الخدمة بطيئة والموظف وقح", "سلبي", "الخدمة"),
    ("الخدمة سيئة جدا", "سلبي", "الخدمة"),
    ("الاسعار مناسبة وممتازة", "ايجابي", "الاسعار"),
    ("الاسعار غاليه وسيئه", "سلبي", "الاسعار"),
    ("الاسعار مرتفعة جدا", "سلبي", "الاسعار"),
    ("الاسعار رخيصة ورائعة", "ايجابي", "الاسعار"),
]
THRESHOLDS = dict(high=0.7, medium=0.55)


def _task_card(task, labels):
    rows = [dict(level=l, accuracy=0.9, n=10) for l in ("high", "medium", "low")]
    return dict(file=f"models/{task}.joblib", labels=labels,
                certainty=dict(thresholds=THRESHOLDS, test_levels=rows),
                test=dict(macro_f1=0.8, per_class_f1={"محايد": 0.4}))


@pytest.fixture
def tiny_models(tmp_path, monkeypatch):
    import sklearn

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    texts = [t for t, _, _ in TEXTS]
    for task, column in (("sentiment", 1), ("aspect", 2)):
        pipe = features.build_pipeline(SPEC).fit(texts, [row[column] for row in TEXTS])
        joblib.dump(pipe, models_dir / f"{task}.joblib")
    card = dict(
        versions={"scikit-learn": sklearn.__version__, "numpy": numpy.__version__, "joblib": joblib.__version__,
                  "python": platform.python_version()},
        tasks=dict(sentiment=_task_card("sentiment", ["ايجابي", "سلبي"]),
                   aspect=_task_card("aspect", ["القهوة", "الخدمة", "الاسعار"])),
        joint_test_accuracy=0.7,
        input=dict(development_length_chars=dict(p99=60)),
    )
    (models_dir / "model_card.json").write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(predict_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(predict_module, "CARD_FILE", models_dir / "model_card.json")
    predict_module.model_card.cache_clear()
    predict_module.load_models.cache_clear()
    yield tmp_path
    predict_module.model_card.cache_clear()
    predict_module.load_models.cache_clear()


def test_prediction_fields_and_levels(tiny_models):
    r = predict("القهوة ممتازة والطعم رائع")
    assert r.sentiment.label == "ايجابي" and r.aspect.label == "القهوة"
    for t, task in ((r.sentiment, "sentiment"), (r.aspect, "aspect")):
        assert 0 < t.score <= 1
        assert t.level == predict_module.certainty_level(t.score, THRESHOLDS)
        if t.level == "high":
            assert t.second_label is None
        else:
            assert t.second_label and t.second_label != t.label and t.second_score <= t.score


def test_same_input_same_output(tiny_models):
    a, b = predict("الخدمة بطيئة"), predict("الخدمة بطيئة")
    assert (a.sentiment, a.aspect) == (b.sentiment, b.aspect)


def test_long_text_and_version_notes(tiny_models):
    assert "long_text" in predict("القهوة جيدة والخدمة سريعة. " * 5).notes
    card = predict_module.model_card()
    assert "library_version" not in predict("القهوة جيدة").notes  # card made with the installed libraries
    card["versions"]["pandas"] = "0.0.0"
    assert "library_version" not in predict("القهوة جيدة").notes  # pandas plays no part at inference
    card["versions"]["python"] = "2.7.18"
    assert "library_version" in predict("القهوة جيدة").notes
    card["versions"]["python"] = predict_module.platform.python_version()
    card["versions"]["numpy"] = "0.0.0"
    assert "library_version" in predict("القهوة جيدة").notes


@pytest.mark.parametrize("text", ["ﻗﻬﻮﺓ ﻣﻤﺘﺎﺯﺓ", "ﺍﻟﺨﺪﻣﺔ ﺳﻴﺌﺔ ﺟﺪﺍ"])
def test_presentation_forms_are_refused(text):
    with pytest.raises(InvalidReview) as err:
        check_review(text)
    assert err.value.code == "unreadable" and err.value.message


@pytest.mark.parametrize("text", ["القهوة ممتازة﻿", "ما شاء ﷲ القهوة ممتازة", "القهوة ممتازة ﺟﺪﺍ", "﴿القهوة ممتازة﴾ ﷼"])
def test_a_few_display_glyphs_beside_plain_text_are_accepted(tiny_models, text):
    assert predict(text)


def test_input_the_models_cannot_read_is_refused(tiny_models):
    # Arabic letters the pipelines never saw: no feature is set, so the output would be the class prior.
    with pytest.raises(InvalidReview) as err:
        predict("ضضضضض")
    assert err.value.code == "unreadable"


@pytest.mark.parametrize("text", ["القهوة ممتازة", "القَهْوَةُ ممتازة", "الخدمة بطيئة جدا 😡", "الاسعار غاليه"])
def test_every_accepted_input_has_features(tiny_models, text):
    check_review(text)
    models = predict_module.load_models()
    for pipe in models.values():
        assert pipe[:-1].transform([text]).nnz > 0
    assert predict(text)


def test_missing_models_raise_file_not_found(tiny_models):
    (tiny_models / "models" / "aspect.joblib").unlink()
    predict_module.load_models.cache_clear()
    with pytest.raises(FileNotFoundError):
        predict("القهوة ممتازة")


def test_cli_prints_utf8_when_stdout_is_a_pipe(tiny_models, capsysbinary, monkeypatch):
    import io

    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252", errors="strict"))
    assert predict_module._main(["القهوة ممتازة والطعم رائع"]) == 0
    sys.stdout.flush()
    assert "ايجابي" in raw.getvalue().decode("utf-8")


def test_app_smoke(tiny_models):
    from streamlit.testing.v1 import AppTest

    app = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception
    at.text_area[0].set_value("الخدمة بطيئة جدا").run()
    assert not at.exception and len(at.subheader) == 2
    at.text_area[0].set_value("hello world").run()
    assert not at.exception and at.warning
    at.text_area[0].set_value("ﺍﻟﺨﺪﻣﺔ ﺳﻴﺌﺔ ﺟﺪﺍ").run()
    assert not at.exception and at.warning


def test_presentation_form_pattern_matches_letters_only():
    import unicodedata

    hits = [c for c in map(chr, range(0xFB50, 0xFF00)) if predict_module.PRESENTATION_FORMS_RE.match(c)]
    assert hits and all(unicodedata.category(c) == "Lo" for c in hits)
