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

# Spelling variants seen in pasted label strings (hamza, ta marbuta, alef maqsura).
_ASPECT_ALT = "ال(?:م[أا]كولات|مشروبات|خدم[ةه]|جلسات|[اأ]سعار)"
_SENTIMENT_ALT = "[اأإ]يجاب[يى]|سلب[يى]|محايد"
_DELIM = "[;:؛]"
_TRAILING = r"[\s.،,!]*$"
# A label string glued to the END of the text. The two-part form ";aspect;sentiment"
# is unambiguous, so spaces around delimiters are allowed. The aspect-only form
# ":aspect" allows no space after the delimiter, which keeps natural "...: الخدمة"
# intact; aspect-only removals are logged for audit.
LABEL_SUFFIX_RE = re.compile(
    rf"\s*{_DELIM}(?:\s*(?P<aspect2>{_ASPECT_ALT})\s*{_DELIM}\s*(?P<sentiment>{_SENTIMENT_ALT})"
    rf"|(?P<aspect1>{_ASPECT_ALT})){_TRAILING}"
)
# Broader scan: a delimiter followed by any label word, anywhere in the text.
LABEL_SCAN_RE = re.compile(rf"{_DELIM}\s*(?:{_ASPECT_ALT}|{_SENTIMENT_ALT})(?![\u0621-\u064a])")
TRANSLATION_MARKER_RE = re.compile(r"\(\s*تمت الترجمة بواسطة Google\s*\.?\s*\)?\s*")
PARAGRAPH_SEP_RE = re.compile(r"(\n[^\S\n]*\n\s*)")
ARABIC_LETTER_RE = re.compile("[\u0621-\u064a\u0671-\u06d3]")
LATIN_LETTER_RE = re.compile("[A-Za-z\u00c0-\u024f\u1e00-\u1eff]")
ASPECT_WORD_RE = re.compile(_ASPECT_ALT)
CONTROL_RE = re.compile(r"[\x00-\x09\x0b-\x1f\x7f-\x9f]")

_SPELLING = str.maketrans({"أ": "ا", "إ": "ا", "ة": "ه", "ى": "ي"})
_CANONICAL = {
    label.translate(_SPELLING): label for label in (*config.ASPECTS, *config.SENTIMENTS)
}


def _canonical(word: str | None) -> str | None:
    return None if word is None else _CANONICAL[word.translate(_SPELLING)]


def strip_label_suffix(text: str) -> tuple[str, str | None, str | None]:
    """Remove a label string glued to the end of the text.

    Returns the repaired text and the embedded (aspect, sentiment), or ``None``
    for each label that was not embedded.
    """
    m = LABEL_SUFFIX_RE.search(text)
    if not m:
        return text, None, None
    return text[: m.start()], _canonical(m["aspect2"] or m["aspect1"]), _canonical(m["sentiment"])


def has_arabic(text: str) -> bool:
    return bool(ARABIC_LETTER_RE.search(text))


def _is_latin_paragraph(par: str) -> bool:
    # Aspect words were pasted into some English translations ("Overالاسعارd."),
    # so they do not count as Arabic content.
    return bool(LATIN_LETTER_RE.search(par)) and not has_arabic(ASPECT_WORD_RE.sub("", par))


def strip_translation(text: str) -> str:
    """Remove platform translations: the Google translation marker and Latin-only
    leading/trailing paragraphs. A paragraph boundary (blank line) is required, as in
    the platform's translation layout, and Arabic text must remain; bilingual lines
    written by the reviewer inside a paragraph are kept."""
    text = TRANSLATION_MARKER_RE.sub("", text)
    parts = PARAGRAPH_SEP_RE.split(text)  # [par, sep, par, sep, ...]
    pars = parts[0::2]
    start, end = 0, len(pars)
    while start < end and _is_latin_paragraph(pars[start]):
        start += 1
    while end > start and _is_latin_paragraph(pars[end - 1]):
        end -= 1
    if (start, end) == (0, len(pars)) or not any(has_arabic(p) for p in pars[start:end]):
        return text
    return "".join(parts[2 * start : 2 * end - 1])


def arabic_letter_count(text: str) -> int:
    return len(ARABIC_LETTER_RE.findall(text))


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
    "delimiter_label_scan": lambda t: bool(LABEL_SCAN_RE.search(t)),
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
    repaired_texts: pd.Series  # rid -> repaired text of every record entering deduplication


def _majority(labels: list[str]) -> str | None:
    """Majority label, or None on a tie for first place."""
    top = Counter(labels).most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


REPAIR_STEPS = (
    ("strip_label_suffix", "label strings glued to the end of the text (CSV-merge artefact)"),
    ("strip_translation", "platform translation marker and Latin-only leading/trailing paragraphs"),
    ("clean_whitespace", "invisible/control characters, repeated spaces, blank lines"),
)
TASKS = ("aspect", "sentiment")


def run_validation(raw: pd.DataFrame, expected_rows: int = config.EXPECTED_ROWS) -> ValidationResult:
    check_schema(raw, expected_rows)
    df = raw.rename(columns=config.RAW_COLUMNS)[["text", "aspect", "sentiment"]].copy()
    df.insert(0, "rid", range(len(df)))  # 0-based data-row index in the source sheet
    source = df.copy()
    profile_raw = profile(df)
    steps = []
    logs = {
        "embedded_label_conflicts": [],
        "aspect_only_suffix_rids": [],
        "translation_removed_arabic_rids": [],
        "label_resolutions": [],
        "duplicate_groups": [],
        "excluded_rids": {},
    }

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
        if emb_aspect is not None and emb_sentiment is None:
            logs["aspect_only_suffix_rids"].append(row.rid)
        for task, emb, official in (("aspect", emb_aspect, row.aspect), ("sentiment", emb_sentiment, row.sentiment)):
            if emb is not None and emb != official:
                logs["embedded_label_conflicts"].append(
                    {"rid": row.rid, "task": task, "embedded": emb, "official": official}
                )
    new = stripped.map(lambda r: r[0])
    record(*REPAIR_STEPS[0], len(df), (new != df.text).sum())
    df["text"] = new

    # 2. Translation segments. Rows where a removed paragraph held Arabic letters other
    #    than the platform marker (aspect words pasted into a translation) are logged.
    new = df.text.map(strip_translation)
    lost = df.text.map(lambda t: arabic_letter_count(TRANSLATION_MARKER_RE.sub("", t))) > new.map(arabic_letter_count)
    logs["translation_removed_arabic_rids"] = df.rid[lost].tolist()
    record(*REPAIR_STEPS[1], len(df), (new != df.text).sum())
    df["text"] = new

    # 3. Whitespace and invisible characters.
    new = df.text.map(clean_whitespace)
    record(*REPAIR_STEPS[2], len(df), (new != df.text).sum())
    df["text"] = new

    # 4. Texts that are empty or contain no Arabic letter after repair.
    keep = df.text.map(has_arabic)
    logs["excluded_rids"]["no_arabic_text"] = df.rid[~keep].tolist()
    record("exclude_no_arabic", "empty or no Arabic letters after repair", len(df), excluded=(~keep).sum())
    df = df[keep]

    # 5. Duplicates on the aggressive key: one representative (lowest rid) per group,
    #    per-task majority label; a tie in either task excludes the whole group.
    repaired = df.set_index("rid").text
    df = df.assign(key=df.text.map(dedup_key))
    rows, copies, ties = [], [], []
    for _, grp in df.groupby("key", sort=False):
        grp = grp.sort_values("rid")
        rep = grp.iloc[0].to_dict()
        rep["dup_group_size"] = len(grp)
        if len(grp) == 1:
            rows.append(rep)
            continue
        rids = grp.rid.tolist()
        resolved = {task: _majority(grp[task].tolist()) for task in TASKS}
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
            ties += rids
            continue
        copies += rids[1:]
        for task, label in resolved.items():
            if label != rep[task]:
                logs["label_resolutions"].append(
                    {"rid": rep["rid"], "task": task, "source": rep[task], "resolved": label, "group_rids": rids}
                )
                rep[task] = label
        rows.append(rep)
    logs["excluded_rids"].update(duplicate_copy=sorted(copies), duplicate_group_label_tie=sorted(ties))
    n_resolved = len({r["rid"] for r in logs["label_resolutions"]})
    record("deduplicate", "one representative per duplicate group; labels by per-task majority",
           len(df), modified=n_resolved, excluded=len(copies))
    record("deduplicate_ties", "duplicate groups whose labels tie in a task", len(df) - len(copies),
           excluded=len(ties))
    data = (
        pd.DataFrame(rows, columns=["rid", "text", "aspect", "sentiment", "dup_group_size"])
        .sort_values("rid")
        .reset_index(drop=True)
    )

    check_integrity(data, source, logs, repaired)
    lineage = {
        "raw_rows": int(len(source)),
        "validated_rows": int(len(data)),
        "excluded_by_reason": {k: len(v) for k, v in logs["excluded_rids"].items()},
        "steps": steps,
    }
    return ValidationResult(data, lineage, logs, profile_raw, profile(data), repaired)


def check_integrity(data: pd.DataFrame, source: pd.DataFrame, logs: dict, repaired: pd.Series) -> None:
    """Hard guarantees on the validated data, recomputed from the source labels and
    the repaired texts (``rid -> text`` before deduplication); any violation aborts."""
    kept = data.rid.tolist()
    if len(set(kept)) != len(kept):
        raise AssertionError("duplicate rids in validated data")
    excluded = [rid for rids in logs["excluded_rids"].values() for rid in rids]
    if len(set(excluded)) != len(excluded) or set(kept) & set(excluded):
        raise AssertionError("a record is excluded twice or both kept and excluded")
    if sorted(kept + excluded) != sorted(source.rid):
        raise AssertionError("kept + excluded records do not partition the source")

    src = source.set_index("rid")
    # Duplicate groups and their outcome are recomputed, not trusted from the log.
    keys = repaired.map(dedup_key)
    recomputed = {tuple(sorted(g.index)) for _, g in keys.groupby(keys) if len(g) > 1}
    if recomputed != {tuple(g["rids"]) for g in logs["duplicate_groups"]}:
        raise AssertionError("logged duplicate groups differ from groups recomputed on the repaired texts")
    if set(logs["excluded_rids"]["no_arabic_text"]) != set(src.index) - set(repaired.index):
        raise AssertionError("no-Arabic exclusions differ from the records missing before deduplication")
    for g in logs["duplicate_groups"]:
        tie = any(_majority(src.loc[g["rids"], task].tolist()) is None for task in TASKS)
        if tie != (g["outcome"] == "excluded_label_tie"):
            raise AssertionError(f"group {g['rids']}: outcome does not match its labels")
    groups = {g["rids"][0]: g["rids"] for g in logs["duplicate_groups"] if g["outcome"] == "kept_representative"}
    resolutions = {(r["rid"], r["task"]): r for r in logs["label_resolutions"]}
    for row in data.itertuples():
        group = groups.get(row.rid, [row.rid])
        if row.dup_group_size != len(group) or min(group) != row.rid:
            raise AssertionError(f"rid {row.rid}: inconsistent duplicate group")
        for task in TASKS:
            expected = _majority(src.loc[group, task].tolist())
            if expected is None or getattr(row, task) != expected:
                raise AssertionError(f"rid {row.rid}: {task} label is not the source (majority) label")
            changed = expected != src.at[row.rid, task]
            if changed != ((row.rid, task) in resolutions):
                raise AssertionError(f"rid {row.rid}: {task} label change not logged exactly")
            if changed and resolutions[(row.rid, task)] != {
                "rid": row.rid, "task": task, "source": src.at[row.rid, task], "resolved": expected, "group_rids": group
            }:
                raise AssertionError(f"rid {row.rid}: {task} resolution record is inconsistent")
    if len(resolutions) != len(logs["label_resolutions"]) or any(rid not in groups for rid, _ in resolutions):
        raise AssertionError("label resolution logged for a record that is not a kept group representative")

    # The broader LABEL_SCAN_RE also matches natural prose ("رأيي: الخدمة"), so it is
    # reported in the quality profile rather than enforced here.
    residual = data.text.map(lambda t: bool(LABEL_SUFFIX_RE.search(t)))
    if residual.any():
        raise AssertionError(f"embedded label strings remain in rids {data.rid[residual].tolist()}")
    if data.text.map(dedup_key).duplicated().any():
        raise AssertionError("duplicates remain after deduplication")
