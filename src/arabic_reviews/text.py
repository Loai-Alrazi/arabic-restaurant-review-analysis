"""Arabic text normalisation: the one implementation used by data validation,
training pipelines and inference.

Every option is explicit and off by default, so ``normalize(text)`` returns the
text unchanged. ``LIGHT`` and ``FULL`` are the presets compared in the
feature-engineering ablations. No option ever deletes a word: negation
particles (``NEGATIONS``) always survive normalisation.
"""

from __future__ import annotations

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
