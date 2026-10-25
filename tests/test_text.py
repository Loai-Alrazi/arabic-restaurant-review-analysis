"""Behaviour of the single text-normalisation module used for training and inference."""

import pytest

from arabic_reviews.text import FULL, LIGHT, NEGATIONS, dedup_key, normalize


def test_light_removes_diacritics_tatweel_and_invisible_marks():
    assert normalize("جمـــيل‏ جدًا", **LIGHT) == "جميل جدا"


def test_light_collapses_letter_runs_of_three_or_more():
    assert normalize("حلووووو مرره", **LIGHT) == "حلو مرره"


def test_light_keeps_legitimate_double_letters():
    assert normalize("الله يعطيكم العافية", **LIGHT) == "الله يعطيكم العافية"


def test_full_unifies_alef_ya_and_ta_marbuta():
    assert normalize("أسعار إيجابية آخر مستوى", **FULL) == "اسعار ايجابيه اخر مستوي"


def test_light_does_not_unify_letters():
    assert normalize("أسعار مستوى", **LIGHT) == "أسعار مستوى"


def test_no_options_returns_text_unchanged():
    assert normalize("أسعار  ممتازة!! 😍") == "أسعار  ممتازة!! 😍"


def test_arabic_indic_digits_map_to_ascii():
    assert normalize("سعره ٢٥ ريال و۳ قطع", digits="ascii") == "سعره 25 ريال و3 قطع"


def test_digit_sequences_can_become_a_single_token():
    assert normalize("سعره ٢٥ ريال", digits="token") == "سعره <NUM> ريال"


def test_symbols_can_be_separated_into_tokens():
    assert normalize("رهيب😍😍!!", symbols="separate") == "رهيب 😍 😍 ! !"


def test_symbols_can_be_removed():
    assert normalize("رهيب😍 جدا!!", symbols="remove") == "رهيب جدا"


@pytest.mark.parametrize("negator", sorted(NEGATIONS))
@pytest.mark.parametrize("preset", [LIGHT, FULL])
def test_every_negation_particle_survives_normalisation(negator, preset):
    out = normalize(f"القهوة {negator} حلوة", **preset, symbols="remove", digits="token")
    assert negator in out.split() or normalize(negator, **preset) in out.split()


def test_elongated_negation_is_restored_not_deleted():
    assert normalize("لاااا ما عجبني", **LIGHT).split()[:2] == ["لا", "ما"]


def test_dedup_key_ignores_spelling_variants_punctuation_and_spacing():
    assert dedup_key("القهوة  رائعة جدًا!!") == dedup_key("القهوه رائعه جدا")


def test_dedup_key_distinguishes_negated_text():
    assert dedup_key("القهوة حلوة") != dedup_key("القهوة مو حلوة")
