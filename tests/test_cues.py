"""Behaviour of the fixed polarity-cue block (version 2)."""

import numpy as np
import pytest

from arabic_reviews import cues
from arabic_reviews.cues import FEATURES, PolarityCues, cue_counts, cue_vector, has_opinion_cue


def _vec(review: str) -> dict[str, float]:
    return dict(zip(FEATURES, cue_vector(review)))


def test_praise_then_complaint_is_mixed():
    v = _vec("ممتاز بس مزدحم")
    assert (v["positive"], v["negative"], v["contrast"], v["mixed"], v["positive_and_contrast"]) == (1, 1, 1, 1, 1)


@pytest.mark.parametrize("review", ["ما عجبني الكيك", "ماعجبني الكيك", "مو حلو ابد", "الطعم غير جيد", "ما كان لذيذ"])
def test_negated_praise_counts_as_a_complaint(review):
    c = cue_counts(review)
    assert c["negated_positive"] == 1 and c["positive"] == 0
    assert _vec(review)["negative"] == 1


@pytest.mark.parametrize("review", ["مافي زحمة", "بدون زحمه"])
def test_negated_complaint_is_neither_praise_nor_complaint(review):
    c = cue_counts(review)
    assert c["negative"] == c["positive"] == c["negated_positive"] == 0


@pytest.mark.parametrize("review", ["الخدمة لا بأس بها", "القهوة جيدة نوعا ما", "مو بطال", "ليس سيء", "القهوة عادية"])
def test_lukewarm_phrases_are_not_read_as_negation(review):
    c = cue_counts(review)
    assert c["lukewarm"] >= 1 and c["negative"] == 0 and c["negated_positive"] == 0


def test_phrases_with_a_negation_particle_keep_their_meaning():
    assert cue_counts("ما شاء الله الشباب ما قصروا")["positive"] == 2
    assert cue_counts("ما أنصح فيه")["negative"] == 1


@pytest.mark.parametrize("review", ["والقهوه ممتازه", "القهوة رائعة", "القهوه رائعه", "روووعه", "بالقهوة الأفضل"])
def test_spelling_variants_clitics_and_elongation_match(review):
    assert cue_counts(review)["positive"] == 1


def test_contrast_with_prefixed_conjunction():
    assert cue_counts("جميل ولكن صغير")["contrast"] == 1
    assert cue_counts("بالرغم من الزحمة")["contrast"] == 1


def test_text_without_any_listed_word_has_no_cue():
    review = "الفرع في شارع التحلية"
    assert not has_opinion_cue(review) and _vec(review)["no_cue"] == 1


def test_counts_are_capped():
    assert _vec("ممتاز رائع لذيذ جميل حلو")["positive"] == cues.CAP


def test_word_lists_do_not_overlap():
    lists = [cues.POSITIVE, cues.NEGATIVE, cues.LUKEWARM, cues.CONTRAST, cues.NEGATORS]
    for i, a in enumerate(lists):
        for b in lists[i + 1:]:
            assert not a & b, sorted(a & b)


def test_transformer_is_stateless_and_shaped():
    block = PolarityCues()
    out = block.fit_transform(["ممتاز بس مزدحم", "سيء"])
    assert out.shape == (2, len(FEATURES))
    assert np.array_equal(PolarityCues().transform(["سيء"]), out[1:])
    assert list(block.get_feature_names_out()) == [f"cue_{f}" for f in FEATURES]
