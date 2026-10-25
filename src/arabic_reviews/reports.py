"""Deterministic writers for the validated dataset, lineage and quality report.

No timestamps are written, so re-running validation reproduces identical files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import config
from .validate import ValidationResult


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def write_outputs(result: ValidationResult, source_sha256: str) -> dict:
    """Write validated.csv, lineage.json and data_quality.md; return their SHA-256."""
    config.VALIDATED_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_csv(config.VALIDATED_FILE, index=False, encoding="utf-8", lineterminator="\n")
    validated_sha = hashlib.sha256(config.VALIDATED_FILE.read_bytes()).hexdigest()

    lineage = {
        "source": {"filename": config.SOURCE["filename"], "doi": config.SOURCE["doi"], "sha256": source_sha256},
        "output": {"file": "data/processed/validated.csv", "rows": len(result.data), "sha256": validated_sha},
        **result.lineage,
        **result.logs,
    }
    _write_text(config.LINEAGE_FILE, json.dumps(lineage, ensure_ascii=False, indent=1) + "\n")
    _write_text(config.QUALITY_REPORT, quality_markdown(result, source_sha256, validated_sha))
    return {
        "validated.csv": validated_sha,
        "lineage.json": hashlib.sha256(config.LINEAGE_FILE.read_bytes()).hexdigest(),
        "data_quality.md": hashlib.sha256(config.QUALITY_REPORT.read_bytes()).hexdigest(),
    }


def _labels_table(p: dict) -> list[str]:
    head = "| aspect \\ sentiment | " + " | ".join(config.SENTIMENTS) + " | total |"
    lines = [head, "|---" * (len(config.SENTIMENTS) + 2) + "|"]
    for a in config.ASPECTS:
        cells = [str(p["joint_counts"][a][s]) for s in config.SENTIMENTS]
        lines.append(f"| {a} | " + " | ".join(cells) + f" | {p['aspect_counts'][a]} |")
    totals = [str(p["sentiment_counts"][s]) for s in config.SENTIMENTS]
    lines.append("| total | " + " | ".join(totals) + f" | {p['rows']} |")
    return lines


def quality_markdown(result: ValidationResult, source_sha256: str, validated_sha256: str) -> str:
    raw, val, lin, logs = result.profile_raw, result.profile_validated, result.lineage, result.logs
    out = [
        "# Data quality report",
        "",
        "Written by `python -m arabic_reviews validate` (deterministic; no timestamps).",
        "",
        f"- Source: `{config.SOURCE['filename']}` — DOI `{config.SOURCE['doi']}` — SHA-256 `{source_sha256}`",
        f"- Output: `data/processed/validated.csv` (local only) — SHA-256 `{validated_sha256}`",
        f"- Rows: raw **{lin['raw_rows']}** → validated **{lin['validated_rows']}**",
        "",
        "## Lineage (repair and exclusion steps)",
        "",
        "| # | step | reason | before | modified | excluded | after |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in lin["steps"]:
        out.append(
            f"| {s['step']} | `{s['name']}` | {s['reason']} | {s['rows_before']} | "
            f"{s['rows_modified']} | {s['rows_excluded']} | {s['rows_after']} |"
        )
    excl = " + ".join(f"{v} {k}" for k, v in lin["excluded_by_reason"].items())
    out += [
        "",
        f"Reconciliation: {lin['raw_rows']} raw = {lin['validated_rows']} kept + {excl}.",
        "",
        "## Quality checks (rows affected)",
        "",
        "| check | raw | validated |",
        "|---|---|---|",
    ]
    for name in raw["rows_with"]:
        out.append(f"| {name} | {raw['rows_with'][name]} | {val['rows_with'][name]} |")
    for kind in ("exact_duplicates", "normalised_duplicates"):
        out.append(
            f"| {kind} (rows in groups / groups) | {raw[kind]['rows_in_groups']} / {raw[kind]['groups']} | "
            f"{val[kind]['rows_in_groups']} / {val[kind]['groups']} |"
        )
    for name, p in (("raw", raw), ("validated", val)):
        t = p["text_length_chars"]
        out.append(f"| text length chars ({name}): min / median / p95 / max | "
                   f"{t['min']} / {t['median']:g} / {t['p95']:g} / {t['max']} | |")
    n_tie_groups = sum(g["outcome"] == "excluded_label_tie" for g in logs["duplicate_groups"])
    conflicts = logs["embedded_label_conflicts"]
    out += [
        "",
        "`negation_particles`, `emoji_or_symbols`, `digits`, `diacritics` and `letter_runs_3plus` are",
        "informative signals and are deliberately **not** repaired here; model-side handling is decided",
        "by ablation inside the training pipeline.",
        "",
        "## Label integrity",
        "",
        f"- Embedded label strings disagreeing with the official label: **{len(conflicts)}** "
        f"({sum(c['task'] == 'aspect' for c in conflicts)} aspect, "
        f"{sum(c['task'] == 'sentiment' for c in conflicts)} sentiment). Logged as label noise; "
        "the official labels are kept.",
        f"- Duplicate groups: **{len(logs['duplicate_groups'])}**; excluded for a label tie: **{n_tie_groups}**; "
        f"labels changed by per-task majority: **{len(logs['label_resolutions'])}**.",
        "- Every other kept record carries its source label unchanged (checked on every run).",
        "- Residual delimiter-wrapped label strings after repair: "
        f"**{val['rows_with']['embedded_label_suffix'] + val['rows_with']['delimited_label_anywhere']}**.",
        "",
        "## Label distribution — raw",
        "",
        *_labels_table(raw),
        "",
        "## Label distribution — validated",
        "",
        *_labels_table(val),
        "",
    ]
    return "\n".join(out)
