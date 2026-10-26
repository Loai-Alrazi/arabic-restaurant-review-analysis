"""Inference, refusals and the demo page on a tiny pipeline built in the test, so they run in CI too.

The real saved pipelines are covered by tests/test_predict.py when they exist locally; here a handful
of synthetic reviews gives two small pipelines with the same structure and a minimal model card.
"""

import json
import platform
import re
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
        test_run=dict(run=1, test_rows=30),
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


def test_models_passed_in_give_the_same_prediction_as_the_saved_ones(tiny_models):
    loaded = {task: joblib.load(tiny_models / "models" / f"{task}.joblib") for task in ("sentiment", "aspect")}
    for text in ("الخدمة بطيئة", "القهوة ممتازة والطعم رائع"):
        saved, passed = predict(text), predict(text, models=loaded)
        assert (saved.sentiment, saved.aspect, saved.notes) == (passed.sentiment, passed.aspect, passed.notes)
    with pytest.raises(InvalidReview):
        predict("great coffee", models=loaded)


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


# --------------------------------------------------------------------------- demo page (AppTest)

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
SERVICE = "الخدمة بطيئة والموظف ما رد علينا"  # EXAMPLES["service"] of the page
PRICES = "الأسعار غالية مقارنة بالكمية"  # EXAMPLES["prices"]
BUILD_V1 = "python -m arabic_reviews final --check"
# Proto fields that hold text a reader sees (labels, bodies, help, values, options).
TEXT_FIELDS = {"body", "label", "help", "placeholder", "value", "text", "caption", "options", "default"}


def _texts(value, name: str = "") -> list[str]:
    from google.protobuf.message import Message

    if isinstance(value, Message):
        return [t for field, v in value.ListFields() for t in _texts(v, field.name)]
    if isinstance(value, str):
        return [value] if name in TEXT_FIELDS else []
    if isinstance(value, bytes) or not hasattr(value, "__iter__"):
        return []
    return [t for v in value for t in _texts(v, name)]  # a repeated field


def element_texts(container) -> list[list[str]]:
    """The visible text of every element inside an AppTest container, whatever its element type."""
    from streamlit.testing.v1.element_tree import Block

    return [_texts(node.proto) for node in container if not isinstance(node, Block)]


def assert_no_numbers(container) -> str:
    """Every element in the container is read (none may be skipped), and none shows a digit or percent sign."""
    texts = element_texts(container)
    assert texts and all(texts), "an element shows no text this check can read; extend TEXT_FIELDS"
    shown = " ".join(t for element in texts for t in element)
    assert not re.search(r"[0-9٠-٩۰-۹%٪]", shown)  # certainty in words only: no score, probability or percentage
    return shown


def analyse(at, text: str):
    """Type a review and press «حلّل المراجعة» (one rerun, as in the browser)."""
    at.text_area[0].set_value(text)
    return at.button(key="analyse").click().run()


def has_result(at) -> bool:
    try:
        at.get_by_key("result")
    except KeyError:
        return False
    return True


def unhamza(s: str) -> str:
    return s.replace("إ", "ا").replace("أ", "ا")  # the page spells some labels with hamza


def assert_shows(at, text: str) -> None:
    """The result card is there and shows the prediction for ``text``."""
    assert has_result(at)
    expected = predict(text)
    for kind, task in (("sentiment", expected.sentiment), ("aspect", expected.aspect)):
        assert unhamza(task.label) in unhamza(at.get_by_key(f"label-{kind}").markdown[0].value)


def test_app_smoke(tiny_models):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and not has_result(at)
    text = "الخدمة بطيئة جدا"
    analyse(at, text)
    assert not at.exception
    assert_shows(at, text)
    card = at.get_by_key("result")
    assert [s.value for s in card.subheader] == ["المشاعر", "الجانب الرئيسي"]
    assert assert_no_numbers(card).count("ثقة النموذج:") == 2
    analyse(at, "hello world")
    assert not at.exception and at.warning and not has_result(at)
    analyse(at, "ﺍﻟﺨﺪﻣﺔ ﺳﻴﺌﺔ ﺟﺪﺍ")
    assert not at.exception and at.warning

    # ?example=<key> fills that example and submits it, so its result is visible on load.
    linked = AppTest.from_file(APP, default_timeout=60)
    linked.query_params["example"] = "service"
    linked.run()
    assert not linked.exception and linked.text_area[0].value == SERVICE
    assert_shows(linked, SERVICE)
    assert_no_numbers(linked.get_by_key("result"))


def test_methodology_uses_card_and_source_provenance(tiny_models, monkeypatch):
    from streamlit.testing.v1 import AppTest

    card_path = tiny_models / "models" / "model_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["tasks"]["sentiment"]["test"]["macro_f1"] = 0.4321
    monkeypatch.setattr(config, "SOURCE", {"creator": "Test Author", "publication_date": "1999-01-01"})

    def method_text(run):
        card["test_run"]["run"] = run
        card_path.write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")
        predict_module.model_card.cache_clear()
        at = AppTest.from_file(APP, default_timeout=60)
        at.query_params["view"] = "method"
        at.run()
        assert not at.exception
        return " ".join(text for element in element_texts(at.tabs[1]) for text in element)

    first = method_text(1)
    assert "0.432" in first and "Author, 1999" in first
    assert "docs/PROBLEM_FRAMING.md" in first and "مرة واحدة" in first

    second = method_text(2)
    assert "مرة واحدة" not in second
    assert "التشغيل المسجّل رقم \u20662\u2069" in second


def test_result_follows_the_submitted_text(tiny_models):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_area[0].set_value("القهوة ممتازة والطعم رائع").run()
    assert not at.exception and not has_result(at)  # typing alone does not analyse
    at.button(key="example-service").click().run()  # an example fills the text and submits it
    assert not at.exception and at.text_area[0].value == SERVICE and at.query_params["example"] == "service"
    assert_shows(at, SERVICE)

    at.text_area[0].set_value("الاسعار مرتفعة جدا").run()  # edited, not submitted: the old result goes
    assert not at.exception and not has_result(at) and "example" not in at.query_params
    assert "تغيّر النص بعد آخر تحليل" in at.get_by_key("result-empty").caption[0].value
    at.button(key="analyse").click().run()
    assert_shows(at, "الاسعار مرتفعة جدا")

    at.button(key="clear").click().run()
    assert not at.exception and at.text_area[0].value == "" and not has_result(at)
    assert "ستظهر النتيجة هنا" in at.get_by_key("result-empty").caption[0].value


def test_example_parameter_is_followed_in_the_same_session(tiny_models):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["example"] = "service"
    at.run()
    assert at.text_area[0].value == SERVICE
    at.query_params["example"] = "prices"  # e.g. back/forward to another example's URL
    at.run()
    assert not at.exception and at.text_area[0].value == PRICES
    assert_shows(at, PRICES)
    at.query_params["example"] = "no-such-example"  # unknown: dropped from the URL, the text stays
    at.run()
    assert not at.exception and "example" not in at.query_params and at.text_area[0].value == PRICES
    at.query_params["example"] = "service"
    at.run()
    assert at.text_area[0].value == SERVICE


def test_unknown_example_on_load_is_dropped(tiny_models):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["example"] = "no-such-example"
    at.run()
    assert not at.exception and "example" not in at.query_params
    assert at.text_area[0].value == "" and not has_result(at)


def test_missing_version_1_models_name_the_version_1_command(tiny_models, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(predict_module, "CARD_FILE_V2", tiny_models / "models" / "model_card_v2.json")  # absent
    (tiny_models / "models" / "sentiment.joblib").unlink()
    predict_module.load_models.cache_clear()
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="example-service").click().run()
    assert not at.exception and at.session_state["version"] == "v1" and not has_result(at)
    assert any(BUILD_V1 in e.value for e in at.error)


def test_presentation_form_pattern_matches_letters_only():
    import unicodedata

    hits = [c for c in map(chr, range(0xFB50, 0xFF00)) if predict_module.PRESENTATION_FORMS_RE.match(c)]
    assert hits and all(unicodedata.category(c) == "Lo" for c in hits)
