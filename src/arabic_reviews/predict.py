"""Inference on new reviews with the saved pipelines.

The saved pipelines contain every text transformation, so the text goes in as
the user typed it. Only input validation happens here, before the models.
Certainty levels are words for bands of the top class probability; the
thresholds are read from ``models/model_card.json``.
"""

from __future__ import annotations

import json
import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import joblib

from . import config

MODELS_DIR = config.ROOT / "models"
CARD_FILE = MODELS_DIR / "model_card.json"

# The development texts have 4-357 characters (99% under 204). A review needs
# at least two Arabic letters, and anything beyond three times the longest
# development text is refused rather than silently truncated.
MIN_ARABIC_LETTERS = 2
MAX_CHARS = 1000
MIN_ARABIC_SHARE = 0.5
TYPICAL_MAX_CHARS = 204

ARABIC_LETTER_RE = re.compile("[ء-غف-يٱ-ۓۺ-ۿ]")
LEVEL_WORDS = {"high": "عالٍ", "medium": "متوسط", "low": "منخفض"}
TASKS = ("sentiment", "aspect")

# Reasons for refusing an input (stable codes, Arabic messages for the page).
REJECTIONS = {
    "empty": "النص فارغ. اكتب مراجعة عن مقهى.",
    "too_short": "النص قصير جدًا؛ يلزم كلمة عربية واحدة على الأقل.",
    "not_arabic": "النص ليس عربيًا في معظمه. النموذج مدرَّب على مراجعات عربية فقط.",
    "too_long": f"النص أطول من {MAX_CHARS} حرفًا. اختصره إلى مراجعة واحدة.",
}


class InvalidReview(ValueError):
    """The input is outside the accepted range; ``code`` is a key of REJECTIONS."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code

    @property
    def message(self) -> str:
        return REJECTIONS[self.code]


@dataclass
class TaskResult:
    label: str
    score: float
    level: str
    second_label: str | None
    second_score: float | None

    @property
    def level_word(self) -> str:
        return LEVEL_WORDS[self.level]


@dataclass
class Prediction:
    sentiment: TaskResult
    aspect: TaskResult
    notes: list[str] = field(default_factory=list)


def check_review(text) -> str:
    """Return the review to score, or raise ``InvalidReview``."""
    if not isinstance(text, str) or not text.strip():
        raise InvalidReview("empty")
    text = unicodedata.normalize("NFC", text).strip()
    if len(text) > MAX_CHARS:
        raise InvalidReview("too_long")
    arabic = len(ARABIC_LETTER_RE.findall(text))
    letters = sum(ch.isalpha() for ch in text)
    if arabic == 0 or letters == 0 or arabic / letters < MIN_ARABIC_SHARE:
        raise InvalidReview("not_arabic")
    if arabic < MIN_ARABIC_LETTERS:
        raise InvalidReview("too_short")
    return text


@lru_cache(maxsize=1)
def model_card() -> dict:
    return json.loads(CARD_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_models() -> dict:
    """Load both saved pipelines; fail clearly when they were not rebuilt."""
    card = model_card()
    models = {}
    for task in TASKS:
        path = config.ROOT / card["tasks"][task]["file"]
        if not path.exists():
            raise FileNotFoundError(f"{path.name} is missing; rebuild it with: python -m arabic_reviews final --check")
        models[task] = joblib.load(path)
    return models


def certainty_level(score: float, thresholds: dict[str, float]) -> str:
    if score >= thresholds["high"]:
        return "high"
    if score >= thresholds["medium"]:
        return "medium"
    return "low"


def _task_result(pipeline, text: str, spec: dict) -> TaskResult:
    proba = pipeline.predict_proba([text])[0]
    order = proba.argsort()[::-1]
    classes = list(pipeline.classes_)
    top = float(proba[order[0]])
    level = certainty_level(top, spec["certainty"]["thresholds"])
    second = level != "high"
    return TaskResult(
        label=classes[order[0]],
        score=top,
        level=level,
        second_label=classes[order[1]] if second else None,
        second_score=float(proba[order[1]]) if second else None,
    )


def predict(text: str) -> Prediction:
    """Score one review. Raises ``InvalidReview`` for input outside the accepted range."""
    text = check_review(text)
    card, models = model_card(), load_models()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sentiment = _task_result(models["sentiment"], text, card["tasks"]["sentiment"])
        aspect = _task_result(models["aspect"], text, card["tasks"]["aspect"])
    notes = []
    if len(text) > TYPICAL_MAX_CHARS:
        notes.append("long_text")
    if len(ARABIC_LETTER_RE.findall(text)) < 4:
        notes.append("very_short")
    return Prediction(sentiment=sentiment, aspect=aspect, notes=notes)


def _main(argv: list[str]) -> int:
    if not argv:
        print('usage: python -m arabic_reviews.predict "review text"')
        return 2
    try:
        r = predict(" ".join(argv))
    except InvalidReview as err:
        print(err.message)
        return 1
    for title, t in (("aspect", r.aspect), ("sentiment", r.sentiment)):
        extra = f" (next: {t.second_label})" if t.second_label else ""
        print(f"{title}: {t.label} | certainty: {t.level_word}{extra}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
