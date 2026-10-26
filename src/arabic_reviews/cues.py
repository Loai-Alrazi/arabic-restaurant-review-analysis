"""Polarity cues (version 2): a small, fixed, hand-written list of Arabic opinion words and
phrases common in cafe reviews (Saudi dialect and Modern Standard Arabic).

Nothing here is learned from data. The lists come from general knowledge of the language and
were written after reading development reviews only (``docs/V2_PLAN.md``). The cue block is a
few dense counts beside the TF-IDF features; it lets a linear model see that a review holds a
positive and a negative judgement at once, which is how most neutral reviews in this dataset
read. Matching uses its own normalised copy of the text (``text.FULL``), so the TF-IDF branch
of a pipeline still sees the text exactly as before.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from . import text


def _forms(words: str) -> frozenset[str]:
    return frozenset(text.normalize(w, **text.FULL) for w in words.split())


def _phrases(*items: str) -> tuple[str, ...]:
    # Longest first, so "الى حد ما" is consumed before "حد ما" could be.
    return tuple(sorted({text.normalize(p, **text.FULL) for p in items}, key=len, reverse=True))


POSITIVE = _forms(
    "ممتاز ممتازة ممتازين رائع رائعة روعة رهيب رهيبة لذيذ لذيذة لذيذين حلو حلوة حلوين جميل جميلة "
    "جيد جيدة كويس كويسة زين زينة مميز مميزة مميزين أفضل أحلى أجمل أطيب أحب أحبه أحبها أحببت عجبني "
    "عجبتني عجبنا يجنن تجنن خيالي خيالية فخم فخمة نظيف نظيفة مريح مريحة هادئ هادئة هادي هادية رايق "
    "رايقة سريع سريعة سريعين لطيف لطيفة لطيفين محترم محترمة محترمين راقي راقية أنصح أنصحكم مناسب "
    "مناسبة رخيص رخيصة مثالي مثالية إبداع مبدع مبدعين يستاهل تستاهل طيب طيبة تحفة شكرا متعاونين "
    "بشوش بشوشين منظم منظمة واسع واسعة نظافة سرعة لذة فنان ألذ أروع مفضل المفضل المفضلة عشق أعشق تمام يتميز "
    "يعجبني احلا معقول معقولة يستحق ممتع ممتعة حبيت"
)
NEGATIVE = _forms(
    "سيء سيئ سيئة سيئين أسوأ أسوء رديء رديئة غالي غالية مرتفع مرتفعة مرتفعه مبالغ مبالغة بطيء بطيئ "
    "بطيئة بطء تأخير تاخير يتأخر يتاخرون متأخر زحمة مزدحم مزدحمة ازدحام زحام يابس يابسة ناشف ناشفة "
    "قاسي محروق محروقة خايس خايسة مقرف مقرفة وسخ وسخة قذر قذرة عفن ضيق ضيقة مزعج مزعجة إزعاج للأسف "
    "خسارة فاشل فاشلة أفشل أغلاط أخطاء مكتوم مكتومة ممل ماسخ ماسخة بايخ بايخة فوضى وقح وقحين مشكلة "
    "مشاكل مخيب بطيئين"
)
LUKEWARM = _forms("عادي عادية عاديين مقبول مقبولة متوسط متوسطة")
CONTRAST = _forms("لكن لاكن لكنه لكنها لاكنه بس إلا أما يعيبه يعيبها يعيب عيبه عيبها رغم بالرغم ماعدا عدا بينما")
NEGATORS = _forms(" ".join(text.NEGATIONS))
POSITIVE_PHRASES = _phrases("ما شاء الله", "غني عن التعريف", "ماشاء الله", "يعطيكم العافية", "ما قصروا", "ماقصروا", "ما قصرو",
                            "ماقصرو", "ما عليه كلام", "ماعليه كلام", "ما عليها كلام")
NEGATIVE_PHRASES = _phrases("مع الأسف", "ما أنصح", "لا أنصح", "ما يستاهل", "مايستاهل", "لا يستحق", "ما تستاهل")
LUKEWARM_PHRASES = _phrases("لا بأس", "لابأس", "لا باس", "لاباس", "نوعا ما", "نوعاً ما", "لحد ما", "الى حد ما",
                            "إلى حد ما", "نص ونص", "مو بطال", "مب بطال", "مش بطال", "ليس سيء", "ليس سيئ",
                            "مو سيء", "مو سيئ", "يمشي الحال", "مو ذاك", "مب ذاك")
CONTRAST_PHRASES = _phrases("مع ان", "مع أن", "غير ان", "غير أن", "ما عدا")

_CLITICS = "وفبلك"
_GLUED_NEGATORS = ("ما", "مو", "مب", "مش")
NEGATION_SCOPE = 2  # a negator reverses a positive cue, or cancels a negative one, among the next two words
CAP = 3
FEATURES = ("positive", "negative", "lukewarm", "contrast", "negated_positive", "mixed", "positive_and_contrast",
            "no_cue")


def _variants(word: str) -> list[str]:
    """The word, without one leading conjunction or preposition, and without the article."""
    out = [word]
    if len(word) > 3 and word[0] in _CLITICS:
        out.append(word[1:])
    for w in list(out):
        if len(w) > 4 and w.startswith("ال"):
            out.append(w[2:])
    return out


def _in(word: str, vocabulary: frozenset[str]) -> bool:
    return any(v in vocabulary for v in _variants(word))


def _take_phrases(norm: str, phrases: tuple[str, ...]) -> tuple[str, int]:
    count = 0
    for p in phrases:
        norm, n = re.subn(rf"(?<!\S)(?:[{_CLITICS}])?{re.escape(p)}(?!\S)", " ", norm)
        count += n
    return norm, count


def cue_counts(review: str) -> dict[str, int]:
    """Counts of each cue type in one review (see ``FEATURES``)."""
    norm = text.normalize(review, **text.FULL, symbols="remove", digits="ascii")
    counts = dict.fromkeys(("positive", "negative", "lukewarm", "contrast", "negated_positive"), 0)
    for key, phrases in (("positive", POSITIVE_PHRASES), ("negative", NEGATIVE_PHRASES),
                         ("lukewarm", LUKEWARM_PHRASES), ("contrast", CONTRAST_PHRASES)):
        norm, n = _take_phrases(norm, phrases)
        counts[key] += n
    words = norm.split()
    scope = 0
    for word in words:
        glued = next((word[len(g):] for g in _GLUED_NEGATORS
                      if word.startswith(g) and len(word) - len(g) >= 3 and _in(word[len(g):], POSITIVE)), None)
        if glued is not None:  # "ماعجبني"
            counts["negated_positive"] += 1
            scope = 0
        elif _in(word, NEGATORS):
            scope = NEGATION_SCOPE
            continue
        elif _in(word, POSITIVE):
            counts["negated_positive" if scope else "positive"] += 1
            scope = 0
        elif _in(word, NEGATIVE):
            if not scope:  # a negated complaint ("مافي زحمة") is no complaint, and not praise either
                counts["negative"] += 1
            scope = 0
        elif _in(word, LUKEWARM):
            counts["lukewarm"] += 1
            scope = 0
        elif _in(word, CONTRAST):
            counts["contrast"] += 1
            scope = 0
        else:
            scope = max(scope - 1, 0)
    return counts


def has_opinion_cue(review: str) -> bool:
    """True when the review holds at least one positive, negative or lukewarm cue."""
    c = cue_counts(review)
    return bool(c["positive"] or c["negative"] or c["lukewarm"] or c["negated_positive"])


def cue_vector(review: str) -> list[float]:
    c = cue_counts(review)
    positive = min(c["positive"], CAP)
    negative = min(c["negative"] + c["negated_positive"], CAP)
    lukewarm = min(c["lukewarm"], CAP)
    contrast = float(c["contrast"] > 0)
    return [
        float(positive), float(negative), float(lukewarm), contrast, float(min(c["negated_positive"], CAP)),
        float(positive > 0 and (negative > 0 or lukewarm > 0)),
        float(positive > 0 and contrast > 0),
        float(positive == negative == lukewarm == 0),
    ]


class PolarityCues(BaseEstimator, TransformerMixin):
    """Stateless dense cue block (``FEATURES``) of the raw review."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([cue_vector(t) for t in X])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"cue_{f}" for f in FEATURES], dtype=object)
