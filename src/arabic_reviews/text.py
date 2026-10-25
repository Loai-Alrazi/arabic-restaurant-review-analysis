"""Arabic text normalisation: the one implementation used by data validation,
training pipelines and inference.

Every option is explicit and off by default, so ``normalize(text)`` returns the
text unchanged. ``LIGHT`` and ``FULL`` are the presets compared in the
feature-engineering ablations. No option ever deletes a word: negation
particles (``NEGATIONS``) always survive normalisation.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Literal

# Negation particles common in Gulf/Saudi and Modern Standard Arabic reviews.
NEGATIONS = frozenset(
    {"لا", "ما", "مو", "مب", "مش", "لم", "لن", "ليس", "ليست", "غير", "بدون", "مافي", "مافيه", "ماني"}
)

# Directional marks, zero-width space, BOM, soft hyphen and Arabic letter mark.
INVISIBLE_RE = re.compile("[­؜​‎‏‪-‮⁦-⁩﻿]")
DIACRITICS_RE = re.compile("[ً-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
LETTER_RUN_RE = re.compile(r"([^\W\d_])\1{2,}")
DIGITS_RE = re.compile(r"\d+")
WHITESPACE_RE = re.compile(r"\s+")
# Emoji joiners, variation selectors and skin-tone modifiers carry no meaning alone.
EMOJI_GLUE_RE = re.compile("[‍︎️\U0001f3fb-\U0001f3ff]")

_ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹"
_DIGIT_TABLE = str.maketrans(_ARABIC_INDIC, "0123456789" * 2)
_NO_MARKS = {ord(ch): None for ch in "".join(map(chr, range(0x064B, 0x0660))) + "ٰ"}
_LETTER_TABLE = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})

LIGHT = dict(
    nfkc=True,
    remove_invisible=True,
    remove_diacritics=True,
    remove_tatweel=True,
    collapse_repeats=True,
    whitespace=True,
)
FULL = dict(LIGHT, unify_letters=True)


def _is_symbol(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "PS"


def normalize(
    text: str,
    *,
    nfkc: bool = False,
    remove_invisible: bool = False,
    remove_diacritics: bool = False,
    remove_tatweel: bool = False,
    collapse_repeats: bool = False,
    unify_letters: bool = False,
    whitespace: bool = False,
    symbols: Literal["keep", "separate", "remove"] = "keep",
    digits: Literal["keep", "ascii", "token"] = "keep",
) -> str:
    """Normalise one review text according to the given options."""
    if nfkc:
        text = unicodedata.normalize("NFKC", text)
    if remove_invisible:
        text = INVISIBLE_RE.sub("", text)
    if remove_diacritics:
        text = DIACRITICS_RE.sub("", text)
    if remove_tatweel:
        text = text.replace(TATWEEL, "")
    if collapse_repeats:
        # Arabic has no legitimate triple letters; elongation ("حلووو") becomes one letter.
        text = LETTER_RUN_RE.sub(r"\1", text)
    if unify_letters:
        text = text.translate(_LETTER_TABLE)
    if symbols != "keep":
        text = EMOJI_GLUE_RE.sub("", text)
        keep = symbols == "separate"
        text = "".join((f" {ch} " if keep else " ") if _is_symbol(ch) else ch for ch in text)
        whitespace = True
    if digits != "keep":
        text = text.translate(_DIGIT_TABLE)
        if digits == "token":
            text = DIGITS_RE.sub(" <NUM> ", text)
            whitespace = True
    if whitespace:
        text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def dedup_key(text: str) -> str:
    """Aggressive key for duplicate detection: spelling variants, symbols, case and
    spacing are ignored, but every word (including negations) is kept."""
    return normalize(text, **FULL, symbols="remove", digits="ascii").lower()


# Frequent Arabic function words. Negation particles are never listed (see NEGATIONS), so
# removing stopwords cannot delete a negation.
STOPWORDS = frozenset(
    {"في", "من", "على", "الى", "إلى", "عن", "مع", "هذا", "هذه", "هذي", "ذلك", "تلك", "هو", "هي",
     "هم", "انا", "أنا", "انت", "أنت", "نحن", "كان", "كانت", "يكون", "التي", "الذي", "الي", "اللي",
     "او", "أو", "ثم", "قد", "كل", "بعض", "هنا", "هناك", "ان", "أن", "إن", "اذا", "إذا", "كما",
     "عند", "حتى", "لكن", "و", "ب", "ل", "ك"}
) - NEGATIONS

# A word keeps its combining marks (diacritics) and tatweel; they are only stripped for lookups.
_WORD_RE = re.compile(r"[\wً-ٰٟـ]+")
_TOKEN_RE = re.compile(r"([\wً-ٰٟـ]+)|([.!?؟…؛;]+)")
NEG_PREFIX = "NEG_"


def _plain(word: str) -> str:
    """Word without diacritics and tatweel, for stopword and negation lookups."""
    return word.replace(TATWEEL, "").translate(_NO_MARKS)


def mark_negation(text: str) -> str:
    """Prefix the word that follows a negation particle with ``NEG_`` (the particle stays;
    a second particle in a row is not prefixed; a sentence boundary ends the scope)."""
    pending = False

    def mark(m: re.Match) -> str:
        nonlocal pending
        word = m.group(1)
        if word is None:  # sentence boundary
            pending = False
            return m.group()
        if _plain(word) in NEGATIONS:
            pending = True
            return word
        if pending:
            pending = False
            return NEG_PREFIX + word
        return word

    return _TOKEN_RE.sub(mark, text)


def remove_stopwords(text: str) -> str:
    """Drop function words in ``STOPWORDS``; every other word, including negations, stays."""
    kept = _WORD_RE.sub(lambda m: "" if _plain(m.group()) in STOPWORDS else m.group(), text)
    return WHITESPACE_RE.sub(" ", kept).strip()


def text_stats(text: str) -> list[float]:
    """Simple surface statistics of the raw review: length, words, '!', '?'/'؟', symbols
    (emoji and pictographs), negation particles."""
    words = _WORD_RE.findall(text)
    return [
        math.log1p(len(text)), math.log1p(len(words)),
        float(text.count("!")), float(text.count("?") + text.count("؟")),
        float(sum(1 for ch in text if unicodedata.category(ch) == "So")),
        float(sum(1 for w in words if _plain(w) in NEGATIONS)),
    ]
