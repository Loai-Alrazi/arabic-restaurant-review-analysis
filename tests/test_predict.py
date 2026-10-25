"""Input validation (always run) and inference on the saved pipelines (run when models/*.joblib exist)."""

import time
from pathlib import Path

import pytest

from arabic_reviews import config
from arabic_reviews.predict import (InvalidReview, LEVEL_WORDS, MAX_CHARS, certainty_level, check_review,
                                    model_card, predict)

HAVE_MODELS = all((config.ROOT / "models" / f"{t}.joblib").exists() for t in ("sentiment", "aspect"))
needs_models = pytest.mark.skipif(not HAVE_MODELS, reason="models/*.joblib not built (python -m arabic_reviews final --check)")


@pytest.mark.parametrize(
    "text, code",
    [
        ("", "empty"),
        ("   \n\t ", "empty"),
        (None, "empty"),
        (123, "empty"),
        ("The coffee was great and the staff were kind", "not_arabic"),
        ("😀😀😀", "not_arabic"),
        ("12345 !!!", "not_arabic"),
        ("ا", "too_short"),
        ("قهوة " * 400, "too_long"),
        ("coffee coffee coffee قهوة", "not_arabic"),
    ],
)
def test_rejected_inputs(text, code):
    with pytest.raises(InvalidReview) as err:
        check_review(text)
    assert err.value.code == code
    assert err.value.message


@pytest.mark.parametrize(
    "text",
    ["القهوة ممتازة", "قَهْوَةٌ لذيييييذة", "قـــهوة حلوة", "ما عجبني الكيك 😡", "كوفي ممتاز coffee", "الجلسه حلوه ٣ نجوم"],
)
def test_accepted_inputs(text):
    assert check_review(text)


def test_max_length_boundary():
    assert check_review("ق" * MAX_CHARS)
    with pytest.raises(InvalidReview):
        check_review("ق" * (MAX_CHARS + 1))


def test_certainty_bands_use_card_thresholds():
    thr = {"high": 0.6, "medium": 0.5}
    assert [certainty_level(s, thr) for s in (0.9, 0.6, 0.59, 0.5, 0.49)] == ["high", "high", "medium", "medium", "low"]


@needs_models
def test_prediction_fields_and_labels():
    card = model_card()
    r = predict("القهوة ممتازة والطعم رائع")
    assert r.sentiment.label in card["tasks"]["sentiment"]["labels"]
    assert r.aspect.label in card["tasks"]["aspect"]["labels"]
    for t in (r.sentiment, r.aspect):
        assert 0 < t.score <= 1 and t.level in LEVEL_WORDS and t.level_word
        if t.level == "high":
            assert t.second_label is None
        else:
            assert t.second_label and t.second_label != t.label and t.second_score <= t.score


@needs_models
def test_obvious_reviews():
    assert predict("الخدمة سيئة جدا والموظف وقح").sentiment.label == "سلبي"
    assert predict("الخدمة ممتازة والموظفين محترمين").sentiment.label == "ايجابي"
    assert predict("الاسعار غاليه مقارنة بالكميه").aspect.label == "الاسعار"


@needs_models
def test_negation_is_not_ignored():
    pos = predict("عجبني الكيك").sentiment
    neg = predict("ما عجبني الكيك").sentiment
    assert pos.label == "ايجابي" and neg.label == "سلبي"


@needs_models
def test_diacritics_tatweel_and_repeats_do_not_flip_the_answer():
    base = predict("القهوة ممتازة").aspect.label
    for variant in ("القَهْوَةُ ممتازة", "القهـــوة ممتازة"):
        r = predict(variant)
        assert r.aspect.label == base and r.sentiment.label == "ايجابي"


@needs_models
def test_edge_inputs_do_not_crash():
    for text in ["ممتاز", "ممتاز 😍😍", "قهوة coffee ممتازة", "ق" * 300, "جيد. " * 150]:
        r = predict(text)
        assert r.sentiment.label and r.aspect.label


@needs_models
def test_long_text_gets_a_note():
    assert "long_text" in predict("القهوة جيدة والخدمة سريعة. " * 12).notes


@needs_models
def test_same_input_same_output():
    a, b = predict("المكان هادئ"), predict("المكان هادئ")
    assert (a.sentiment, a.aspect) == (b.sentiment, b.aspect)


@needs_models
def test_cpu_latency_for_one_review():
    predict("تجربة")  # load models once
    start = time.perf_counter()
    for _ in range(20):
        predict("القهوة ممتازة والخدمة سريعة والمكان نظيف")
    assert (time.perf_counter() - start) / 20 < 0.1


@needs_models
def test_app_smoke():
    from streamlit.testing.v1 import AppTest

    app = Path(config.ROOT / "app" / "streamlit_app.py")
    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception
    at.text_area[0].set_value("الخدمة بطيئة جدا").run()
    assert not at.exception and len(at.subheader) == 2
    at.text_area[0].set_value("hello world").run()
    assert not at.exception and at.warning


@needs_models
def test_levels_follow_the_card_thresholds():
    card = model_card()
    r = predict("القهوة ممتازة والطعم رائع")
    for task in ("sentiment", "aspect"):
        t = getattr(r, task)
        assert t.level == certainty_level(t.score, card["tasks"][task]["certainty"]["thresholds"])
