"""Schema/quality validation and conservative, logged repair of the source file.

Repair fixes defects of this particular file (label strings pasted into the text,
platform translations, invisible characters, duplicates). It is a data step with
lineage, separate from the model-side normalisation in ``text.py``.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from . import config
from .text import DIACRITICS_RE, INVISIBLE_RE, LETTER_RUN_RE, NEGATIONS, TATWEEL, dedup_key

_ASPECT_ALT = "المأكولات|المشروبات|الخدمة|الجلسات|ال[اأ]سعار"
_SENTIMENT_ALT = "[اإ]يجابي|سلب[يى]|محايد"
# A label string glued to the END of the text: ";aspect;sentiment", ":aspect", ...
# No space is allowed after the delimiter, which keeps natural "...: الخدمة" intact.
LABEL_SUFFIX_RE = re.compile(
    rf"\s*[;:؛](?P<aspect>{_ASPECT_ALT})(?:[;:؛](?P<sentiment>{_SENTIMENT_ALT}))?[\s.]*$"
)
# Scan for any delimiter-wrapped label anywhere in the text (must find nothing after repair).
LABEL_ANYWHERE_RE = re.compile(
    rf"[;:؛](?:{_ASPECT_ALT})(?:[;:؛]|\s*$)|[;:؛](?:{_SENTIMENT_ALT})\s*$"
)
TRANSLATION_MARKER_RE = re.compile(r"\(\s*تمت الترجمة بواسطة Google\s*\.?\s*\)?\s*")
ARABIC_LETTER_RE = re.compile("[ء-يٱ-ۓ]")
LATIN_LETTER_RE = re.compile("[A-Za-zÀ-ɏḀ-ỿ]")
ASPECT_WORD_RE = re.compile(_ASPECT_ALT)
CONTROL_RE = re.compile(r"[\x00-\x09\x0b-\x1f\x7f-\x9f]")


def _canonical_aspect(word: str) -> str:
    return "الاسعار" if word in ("الأسعار", "الاسعار") else word


def _canonical_sentiment(word: str | None) -> str | None:
    if word is None:
        return None
    return {"إيجابي": "ايجابي", "سلبى": "سلبي"}.get(word, word)


def strip_label_suffix(text: str) -> tuple[str, str | None, str | None]:
    """Remove a label string glued to the end of the text.

    Returns the repaired text and the embedded (aspect, sentiment), or ``None``
    for each label that was not embedded.
    """
    m = LABEL_SUFFIX_RE.search(text)
    if not m:
        return text, None, None
    return text[: m.start()], _canonical_aspect(m["aspect"]), _canonical_sentiment(m["sentiment"])


def has_arabic(text: str) -> bool:
    return bool(ARABIC_LETTER_RE.search(text))


def _is_latin_line(line: str) -> bool:
    # Aspect words were pasted into some English translations ("Overالاسعارd."),
    # so they do not count as Arabic content.
    return bool(LATIN_LETTER_RE.search(line)) and not has_arabic(ASPECT_WORD_RE.sub("", line))


def strip_translation(text: str) -> str:
    """Remove platform translations: the Google translation marker and leading or
    trailing Latin-only lines (with the blank lines around them), but only when
    Arabic text remains."""
    text = TRANSLATION_MARKER_RE.sub("", text)
    lines = text.split("\n")
    if len(lines) < 2:
        return text
    start, end = 0, len(lines)
    # Advance past leading Latin/blank lines; cut there only if a Latin line was skipped.
    i = 0
    while i < end and (_is_latin_line(lines[i]) or not lines[i].strip()):
        if _is_latin_line(lines[i]):
            start = i + 1
        i += 1
    j = end
    while j > start and (_is_latin_line(lines[j - 1]) or not lines[j - 1].strip()):
        if _is_latin_line(lines[j - 1]):
            end = j - 1
        j -= 1
    if (start, end) == (0, len(lines)) or not any(has_arabic(line) for line in lines[start:end]):
        return text
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end])


def clean_whitespace(text: str) -> str:
    """Drop invisible/control characters, collapse spaces and drop blank lines."""
    text = INVISIBLE_RE.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = CONTROL_RE.sub(" ", text)
    lines = (re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------- schema


class SchemaError(ValueError):
    """The source frame does not have the documented structure."""


def check_schema(raw: pd.DataFrame, expected_rows: int = config.EXPECTED_ROWS) -> None:
    missing = set(config.RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise SchemaError(f"missing columns: {sorted(missing)}")
    if len(raw) != expected_rows:
        raise SchemaError(f"expected {expected_rows} rows, found {len(raw)}")
    for col in config.RAW_COLUMNS:
        if raw[col].isna().any():
            raise SchemaError(f"null values in {col}")
        if not raw[col].map(lambda v: isinstance(v, str)).all():
            raise SchemaError(f"non-text values in {col}")
    for col, allowed in (("Attribute", config.ASPECTS), ("Result", config.SENTIMENTS)):
        unknown = set(raw[col]) - set(allowed)
        if unknown:
            raise SchemaError(f"unknown labels in {col}: {sorted(unknown)}")


# ---------------------------------------------------------------- quality profile

SPREADSHEET_REF_RE = re.compile(r"[A-Z]{1,2}\d+:[A-Z]{1,2}\d+")


def _has_negation(text: str) -> bool:
    return any(tok in NEGATIONS for tok in re.findall(r"\w+", text))


QUALITY_CHECKS = {
    "embedded_label_suffix": lambda t: bool(LABEL_SUFFIX_RE.search(t)),
    "delimited_label_anywhere": lambda t: bool(LABEL_ANYWHERE_RE.search(t)),
    "translation_marker": lambda t: bool(TRANSLATION_MARKER_RE.search(t)),
    "latin_letters": lambda t: bool(LATIN_LETTER_RE.search(t)),
    "no_arabic_letters": lambda t: not has_arabic(t),
    "invisible_characters": lambda t: bool(INVISIBLE_RE.search(t)),
    "control_characters": lambda t: bool(CONTROL_RE.search(t)),
    "diacritics": lambda t: bool(DIACRITICS_RE.search(t)),
    "tatweel": lambda t: TATWEEL in t,
    "letter_runs_3plus": lambda t: bool(LETTER_RUN_RE.search(t)),
    "emoji_or_symbols": lambda t: any(unicodedata.category(c) == "So" for c in t),
    "digits": lambda t: any(c.isdigit() for c in t),
    "negation_particles": _has_negation,
    "spreadsheet_cell_reference": lambda t: bool(SPREADSHEET_REF_RE.search(t)),
}


def _duplicate_stats(keys: pd.Series) -> dict:
    sizes = keys.value_counts()
    dup = sizes[sizes > 1]
    return {"rows_in_groups": int(dup.sum()), "groups": int(len(dup)), "unique_texts": int(len(sizes))}


def profile(df: pd.DataFrame) -> dict:
    """Quality profile of a frame with columns text/aspect/sentiment."""
    lengths = df.text.str.len()
    return {
        "rows": int(len(df)),
        "aspect_counts": {a: int((df.aspect == a).sum()) for a in config.ASPECTS},
        "sentiment_counts": {s: int((df.sentiment == s).sum()) for s in config.SENTIMENTS},
        "joint_counts": {
            a: {s: int(((df.aspect == a) & (df.sentiment == s)).sum()) for s in config.SENTIMENTS}
            for a in config.ASPECTS
        },
        "text_length_chars": {
            "min": int(lengths.min()),
            "median": float(lengths.median()),
            "p95": float(lengths.quantile(0.95)),
            "max": int(lengths.max()),
        },
        "rows_with": {name: int(df.text.map(fn).sum()) for name, fn in QUALITY_CHECKS.items()},
        "exact_duplicates": _duplicate_stats(df.text),
        "normalised_duplicates": _duplicate_stats(df.text.map(dedup_key)),
    }


# ---------------------------------------------------------------- validation run


@dataclass
class ValidationResult:
    data: pd.DataFrame
    lineage: dict
    logs: dict
    profile_raw: dict
    profile_validated: dict


def _majority(labels: list[str]) -> str | None:
    """Majority label, or None on a tie for first place."""
    top = Counter(labels).most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


REPAIR_STEPS = (
    ("strip_label_suffix", "label strings glued to the end of the text (CSV-merge artefact)"),
    ("strip_translation", "platform translation marker and Latin-only leading/trailing lines"),
    ("clean_whitespace", "invisible/control characters, repeated spaces, blank lines"),
)


def run_validation(raw: pd.DataFrame, expected_rows: int = config.EXPECTED_ROWS) -> ValidationResult:
    check_schema(raw, expected_rows)
    df = raw.rename(columns=config.RAW_COLUMNS)[["text", "aspect", "sentiment"]].copy()
    df.insert(0, "rid", range(len(df)))  # 0-based data-row index in the source sheet
    source = df.copy()
    profile_raw = profile(df)
    steps, logs = [], {"embedded_label_conflicts": [], "label_resolutions": [], "duplicate_groups": []}

    def record(name, reason, before, modified=0, excluded=0):
        steps.append(
            {
                "step": len(steps) + 1,
                "name": name,
                "reason": reason,
                "rows_before": int(before),
                "rows_modified": int(modified),
                "rows_excluded": int(excluded),
                "rows_after": int(before) - int(excluded),
            }
        )

    # 1. Embedded label strings: removed; disagreements with the official label logged only.
    stripped = df.text.map(strip_label_suffix)
    for row, (_, emb_aspect, emb_sentiment) in zip(df.itertuples(), stripped):
        for task, emb, official in (("aspect", emb_aspect, row.aspect), ("sentiment", emb_sentiment, row.sentiment)):
            if emb is not None and emb != official:
                logs["embedded_label_conflicts"].append(
                    {"rid": row.rid, "task": task, "embedded": emb, "official": official}
                )
    new = stripped.map(lambda r: r[0])
    record(*REPAIR_STEPS[0], len(df), (new != df.text).sum())
    df["text"] = new

    # 2-3. Translation segments, whitespace and invisible characters.
    for (name, reason), fn in zip(REPAIR_STEPS[1:], (strip_translation, clean_whitespace)):
        new = df.text.map(fn)
        record(name, reason, len(df), (new != df.text).sum())
        df["text"] = new

    # 4. Texts that are empty or contain no Arabic letter after repair.
    keep = df.text.map(has_arabic)
    excluded = {"no_arabic_text": int((~keep).sum())}
    logs["no_arabic_text_rids"] = df.rid[~keep].tolist()
    record("exclude_no_arabic", "empty or no Arabic letters after repair", len(df), excluded=(~keep).sum())
    df = df[keep]

    # 5. Duplicates on the aggressive key: one representative (lowest rid) per group,
    #    per-task majority label; a tie in either task excludes the whole group.
    df = df.assign(key=df.text.map(dedup_key))
    rows, n_copy, n_tie = [], 0, 0
    for _, grp in df.groupby("key", sort=False):
        grp = grp.sort_values("rid")
        rep = grp.iloc[0].to_dict()
        rep["dup_group_size"] = len(grp)
        if len(grp) == 1:
            rows.append(rep)
            continue
        rids = grp.rid.tolist()
        resolved = {task: _majority(grp[task].tolist()) for task in ("aspect", "sentiment")}
        tie = None in resolved.values()
        logs["duplicate_groups"].append(
            {
                "rids": rids,
                "aspect_votes": dict(sorted(Counter(grp.aspect).items())),
                "sentiment_votes": dict(sorted(Counter(grp.sentiment).items())),
                "outcome": "excluded_label_tie" if tie else "kept_representative",
            }
        )
        if tie:
            n_tie += len(grp)
            continue
        n_copy += len(grp) - 1
        for task, label in resolved.items():
            if label != rep[task]:
                logs["label_resolutions"].append(
                    {"rid": rep["rid"], "task": task, "source": rep[task], "resolved": label, "group_rids": rids}
                )
                rep[task] = label
        rows.append(rep)
    excluded.update(duplicate_copy=n_copy, duplicate_group_label_tie=n_tie)
    record("deduplicate", "one representative per duplicate group (copies)", len(df), excluded=n_copy)
    record("deduplicate_ties", "duplicate groups whose labels tie in a task", len(df) - n_copy, excluded=n_tie)
    data = (
        pd.DataFrame(rows, columns=["rid", "text", "aspect", "sentiment", "dup_group_size"])
        .sort_values("rid")
        .reset_index(drop=True)
    )

    _check_integrity(data, source, logs, excluded)
    lineage = {
        "raw_rows": int(len(source)),
        "validated_rows": int(len(data)),
        "excluded_by_reason": excluded,
        "steps": steps,
    }
    return ValidationResult(data, lineage, logs, profile_raw, profile(data))


def _check_integrity(data: pd.DataFrame, source: pd.DataFrame, logs: dict, excluded: dict) -> None:
    """Hard guarantees on the validated data; any violation aborts the run."""
    if len(source) != len(data) + sum(excluded.values()):
        raise AssertionError("row counts do not reconcile")
    resolved = {(r["rid"], r["task"]): r["resolved"] for r in logs["label_resolutions"]}
    src = source.set_index("rid")
    for row in data.itertuples():
        for task in ("aspect", "sentiment"):
            expected = resolved.get((row.rid, task), src.at[row.rid, task])
            if getattr(row, task) != expected:
                raise AssertionError(f"label changed for rid {row.rid} ({task})")
    residual = data.text.map(lambda t: bool(LABEL_SUFFIX_RE.search(t) or LABEL_ANYWHERE_RE.search(t)))
    if residual.any():
        raise AssertionError(f"embedded label strings remain in rids {data.rid[residual].tolist()}")
    if data.text.map(dedup_key).duplicated().any():
        raise AssertionError("duplicates remain after deduplication")
