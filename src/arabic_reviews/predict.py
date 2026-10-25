"""Inference on new reviews with the saved pipelines.

The saved pipelines contain every text transformation, so the text goes in as
the user typed it. Only input validation happens here, before the models.
Certainty levels are words for bands of the top class probability; the
thresholds are read from the model card of the chosen version:
``models/model_card.json`` (version 1, one threshold pair per task) or
``models/model_card_v2.json`` (version 2, thresholds per predicted class,
where a level may not exist for a class).
"""

from __future__ import annotations

import json
import platform
import re
import sys
import unicodedata
import warnings
from importlib import metadata
from dataclasses import dataclass, field
from functools import lru_cache

import joblib

from . import config

MODELS_DIR = config.ROOT / "models"
CARD_FILE = MODELS_DIR / "model_card.json"
CARD_FILE_V2 = MODELS_DIR / "model_card_v2.json"
VERSIONS = ("v1", "v2")

# The development texts have 4-357 characters (99% under 204). A review needs
# at least two Arabic letters, and anything beyond 1000 characters (about
# three times the longest development text) is refused rather than silently
# truncated.
MIN_ARABIC_LETTERS = 2
MAX_CHARS = 1000
MIN_ARABIC_SHARE = 0.5

ARABIC_LETTER_RE = re.compile("[ء-غف-يٱ-ۓۺ-ۿ]")
# Arabic Presentation Forms-A/B letters: display-only glyphs that the saved pipelines (no normalisation) do not
# map to the plain letters, so they reach the model as unknown characters. U+FEFF (zero-width mark) is not one.
PRESENTATION_FORMS_RE = re.compile(
    r"[\ufb50-\ufbb1\ufbd3-\ufd3d\ufd50-\ufd8f\ufd92-\ufdc7\ufdf0-\ufdfb\ufe70-\ufe74\ufe76-\ufefc]")
# Library versions that decide how a saved pipeline loads and computes (pandas only builds the training tables).
RUNTIME_LIBRARIES = ("scikit-learn", "numpy", "joblib")


def _arabic_count(text: str) -> int:
    """Arabic letters, also when written in presentation forms (counted on an NFKC copy)."""
    return len(ARABIC_LETTER_RE.findall(unicodedata.normalize("NFKC", text)))


LEVEL_WORDS = {"high": "عالٍ", "medium": "متوسط", "low": "منخفض"}
TASKS = ("sentiment", "aspect")

# Reasons for refusing an input (stable codes, Arabic messages for the page).
REJECTIONS = {
    "empty": "النص فارغ. اكتب مراجعة عن مقهى.",
    "too_short": "النص قصير جدًا؛ يلزم حرفان عربيان على الأقل.",
    "not_arabic": "النص ليس عربيًا في معظمه. النموذج مدرَّب على مراجعات عربية فقط.",
    "too_long": f"النص أطول من {MAX_CHARS} حرفًا. اختصره إلى مراجعة واحدة.",
    "unreadable": "لا يستطيع النموذج قراءة هذا النص (أشكال عرض عربية أو حروف غير معروفة). الصق المراجعة كنص عادي.",
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
    version: str = "v1"


def check_review(text) -> str:
    """Return the review to score, or raise ``InvalidReview``."""
    if not isinstance(text, str) or not text.strip():
        raise InvalidReview("empty")
    text = text.strip()
    if len(text) > MAX_CHARS:
        raise InvalidReview("too_long")
    if len(PRESENTATION_FORMS_RE.findall(text)) > len(ARABIC_LETTER_RE.findall(text)):
        raise InvalidReview("unreadable")  # mostly display forms; a few shaped letters beside plain text are harmless
    arabic = _arabic_count(text)
    letters = sum(ch.isalpha() for ch in unicodedata.normalize("NFKC", text))
    if arabic == 0 or letters == 0 or arabic / letters < MIN_ARABIC_SHARE:
        raise InvalidReview("not_arabic")
    if arabic < MIN_ARABIC_LETTERS:
        raise InvalidReview("too_short")
    return text


REBUILD = {"v1": "python -m arabic_reviews final --check", "v2": "python -m arabic_reviews v2-build"}


def _card_file(version: str):
    if version not in VERSIONS:
        raise ValueError(f"unknown model version: {version}")
    return CARD_FILE if version == "v1" else CARD_FILE_V2


@lru_cache(maxsize=2)
def _card(version: str) -> dict:
    return json.loads(_card_file(version).read_text(encoding="utf-8"))


def model_card(version: str = "v1") -> dict:
    """The model card of a version (cached: ``model_card()`` and ``model_card("v1")`` are one object)."""
    return _card(version)


@lru_cache(maxsize=2)
def _models(version: str) -> dict:
    card = model_card(version)
    models = {}
    for task in TASKS:
        path = config.ROOT / card["tasks"][task]["file"]
        if not path.exists():
            raise FileNotFoundError(f"{path.name} is missing; rebuild it with: {REBUILD[version]}")
        models[task] = joblib.load(path)
    return models


def load_models(version: str = "v1") -> dict:
    """Load both saved pipelines of a version (cached); fail clearly when they were not rebuilt."""
    return _models(version)


model_card.cache_clear = _card.cache_clear
load_models.cache_clear = _models.cache_clear


def available_versions() -> list[str]:
    """Versions whose model card and saved pipelines are all present."""
    out = []
    for version in VERSIONS:
        path = _card_file(version)
        if path.exists():
            card = json.loads(path.read_text(encoding="utf-8"))
            if all((config.ROOT / card["tasks"][task]["file"]).exists() for task in TASKS):
                out.append(version)
    return out


def certainty_level(score: float, thresholds: dict[str, float | None]) -> str:
    """Level of a score; a level whose threshold is None does not exist (version 2, per class)."""
    if thresholds.get("high") is not None and score >= thresholds["high"]:
        return "high"
    if thresholds.get("medium") is not None and score >= thresholds["medium"]:
        return "medium"
    return "low"


def thresholds_for(spec: dict, label: str) -> dict[str, float | None]:
    certainty = spec["certainty"]
    if "thresholds_by_class" in certainty:
        return certainty["thresholds_by_class"][label]
    return certainty["thresholds"]


def library_mismatch(card: dict) -> bool:
    """True when an installed runtime library (or the Python minor version) differs from the card's."""
    for name in RUNTIME_LIBRARIES:
        try:
            same = metadata.version(name) == card["versions"][name]
        except (metadata.PackageNotFoundError, KeyError):
            same = False
        if not same:
            return True
    python = lambda v: ".".join(v.split(".")[:2])
    return python(platform.python_version()) != python(card["versions"].get("python", platform.python_version()))


def _task_result(pipeline, text: str, spec: dict) -> TaskResult:
    proba = pipeline.predict_proba([text])[0]
    order = proba.argsort()[::-1]
    classes = list(pipeline.classes_)
    top = float(proba[order[0]])
    level = certainty_level(top, thresholds_for(spec, classes[order[0]]))
    second = level != "high"
    return TaskResult(
        label=classes[order[0]],
        score=top,
        level=level,
        second_label=classes[order[1]] if second else None,
        second_score=float(proba[order[1]]) if second else None,
    )


def predict(text: str, models: dict | None = None, version: str = "v1") -> Prediction:
    """Score one review. Raises ``InvalidReview`` for input outside the accepted range.

    ``models`` maps each task to a fitted pipeline of the saved configuration; by default the saved
    pipelines of ``version`` are loaded. Thresholds and notes always come from that version's model card."""
    text = check_review(text)
    card = model_card(version)
    models = load_models(version) if models is None else models
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if any(pipeline[:-1].transform([text]).nnz == 0 for pipeline in models.values()):
            raise InvalidReview("unreadable")  # no feature set: the output would only be the class prior
        sentiment = _task_result(models["sentiment"], text, card["tasks"]["sentiment"])
        aspect = _task_result(models["aspect"], text, card["tasks"]["aspect"])
    notes = []
    if len(text) > card["input"]["development_length_chars"]["p99"]:
        notes.append("long_text")
    if _arabic_count(text) < 4:
        notes.append("very_short")
    if library_mismatch(card):
        notes.append("library_version")
    return Prediction(sentiment=sentiment, aspect=aspect, notes=notes, version=version)


def _main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # a piped stdout on Windows is not UTF-8
    version = "v1"
    if argv[:1] == ["--version"] and len(argv) > 1:
        version, argv = argv[1], argv[2:]
    if not argv or version not in VERSIONS:
        print('usage: python -m arabic_reviews.predict [--version v1|v2] "review text"')
        return 2
    try:
        r = predict(" ".join(argv), version=version)
    except InvalidReview as err:
        print(err.message)
        return 1
    for title, t in (("aspect", r.aspect), ("sentiment", r.sentiment)):
        extra = f" (next: {t.second_label})" if t.second_label else ""
        print(f"{title}: {t.label} | certainty: {t.level_word}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
