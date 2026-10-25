"""Deterministic writers for the validated dataset, lineage and quality report.

No timestamps are written, so re-running validation reproduces identical files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from . import config
from .validate import ValidationResult


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def write_outputs(
    result: ValidationResult,
    source_sha256: str,
    validated_file: Path = config.VALIDATED_FILE,
    lineage_file: Path = config.LINEAGE_FILE,
    quality_report: Path = config.QUALITY_REPORT,
) -> dict:
    """Write validated.csv, lineage.json and data_quality.md; return their SHA-256."""
    validated_file.parent.mkdir(parents=True, exist_ok=True)
    result.data.to_csv(validated_file, index=False, encoding="utf-8", lineterminator="\n")
    validated_sha = hashlib.sha256(validated_file.read_bytes()).hexdigest()

    lineage = {
        "source": {"filename": config.SOURCE["filename"], "doi": config.SOURCE["doi"], "sha256": source_sha256},
        "output": {"file": "data/processed/validated.csv", "rows": len(result.data), "sha256": validated_sha},
        **result.lineage,
        **result.logs,
    }
    _write_text(lineage_file, json.dumps(lineage, ensure_ascii=False, indent=1) + "\n")
    _write_text(quality_report, quality_markdown(result, source_sha256, validated_sha))
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (validated_file, lineage_file, quality_report)
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
        "- Every other kept record carries its source label unchanged; labels, group sizes and the",
        "  kept/excluded partition are recomputed from the source on every run.",
        f"- Aspect-only suffixes removed (`:aspect`, audited in `lineage.json`): "
        f"**{len(logs['aspect_only_suffix_rids'])}**.",
        f"- Translation paragraphs removed that contained pasted Arabic aspect words (audited): "
        f"**{len(logs['translation_removed_arabic_rids'])}**.",
        "- Residual label strings after repair (end-anchored pattern / delimiter scan anywhere): "
        f"**{val['rows_with']['embedded_label_suffix']} / {val['rows_with']['delimiter_label_scan']}**.",
        "- Known residual noise left as is: Latin words inside Arabic sentences or bilingual lines"
        " written by the reviewer, and one spreadsheet cell reference inside a text.",
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


def write_split_outputs(data, result, validated_sha256: str) -> dict:
    """Write split_v1.csv, near_duplicates.csv and split.md (published) plus the local
    dev.csv/test.csv text files; return the SHA-256 of every file written."""
    from . import split as sp

    a = result.assignment
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    a.to_csv(config.SPLIT_FILE, index=False, lineterminator="\n")
    result.pairs.to_csv(config.NEAR_DUPLICATES_FILE, index=False, lineterminator="\n")
    full = data.drop(columns=["dup_group_size"], errors="ignore").merge(a, on="rid", validate="one_to_one")
    dev = full[full.partition == "dev"].drop(columns=["partition"])
    test = full[full.partition == "test"].drop(columns=["partition", *sp.CV_COLUMNS])
    dev.to_csv(config.DEV_FILE, index=False, encoding="utf-8", lineterminator="\n")
    test.to_csv(config.TEST_FILE, index=False, encoding="utf-8", lineterminator="\n")
    checks = sp.leakage_checks(data, a, result.pairs)
    props = sp.proportions(data, a)
    _write_text(config.SPLIT_REPORT, split_markdown(full, result, checks, props, validated_sha256))
    files = (config.SPLIT_FILE, config.NEAR_DUPLICATES_FILE, config.SPLIT_REPORT, config.DEV_FILE, config.TEST_FILE)
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def split_markdown(full, result, checks: dict, props, validated_sha256: str) -> str:
    from . import split as sp

    sizes = full.groupby("group_id").size()
    size_counts = sizes.value_counts().sort_index()
    pairs = result.pairs
    out = [
        "# Split report",
        "",
        "Written by `python -m arabic_reviews split` (deterministic; no timestamps).",
        "",
        f"- Input: `data/processed/validated.csv` — SHA-256 `{validated_sha256}` — {len(full)} rows",
        "- Output: `data/splits/split_v1.csv` (published, no text: `rid, group_id, partition, "
        + ", ".join(sp.CV_COLUMNS) + "`), `reports/near_duplicates.csv` (pairs by rid);"
        " `data/processed/dev.csv` and `test.csv` stay local.",
        "",
        "## 1. Near-duplicate grouping (label-free)",
        "",
        f"Keys: `text.dedup_key`. Similarity: cosine on character 3–5-gram TF-IDF (`char_wb`, sublinear tf),"
        f" fitted on all validated texts — a quality step that never sees labels. A pair is linked when"
        f" cosine ≥ **{sp.SIM_THRESHOLD:.2f}**, or when one key of at least **{sp.CONTAIN_MIN_WORDS} words**"
        f" occurs inside another. Groups are connected components (union-find); `group_id` is the smallest"
        f" `rid` in the group.",
        "",
        "Pairs per similarity band (all row pairs, before the containment rule):",
        "",
        "| cosine | " + " | ".join(result.similarity_bands) + " |",
        "|---" * (len(result.similarity_bands) + 1) + "|",
        "| pairs | " + " | ".join(str(v) for v in result.similarity_bands.values()) + " |",
        "",
        "Threshold choice (from reading sampled pairs in every band from 0.60 to 1.00):",
        "",
        "- **≥ 0.80**: the same review copied with a few words cut or added (for example rids 681/785,"
        " 436/1336, 446/1346, 569/607) or the same short stock phrase (\"quiet place\" / \"quiet place, very\")."
        " Linking stock phrases is conservative: a model cannot score by recognising a phrase it saw in training.",
        "- **0.70–0.80**: mostly different reviews sharing vocabulary, some with different meaning or label"
        " (for example rids 248/249 differ only in *coffee* vs *sweets*), mixed with a few truncated copies"
        " (rid 364 is the tail of rids 407 and 1311). Grouping by cosine here would merge distinct reviews.",
        f"- The containment rule recovers truncated copies whose cosine falls below the threshold"
        f" ({int((pairs.contained & (pairs.cosine < sp.SIM_THRESHOLD)).sum())} pairs, for example rids 344/375,"
        f" 438/1338, 1795/2552). It requires {sp.CONTAIN_MIN_WORDS}+ words; shorter keys (\"good coffee\")"
        " are common phrases rather than copies.",
        "",
        f"Result: {len(pairs)} linked pairs ({int((pairs.cosine >= sp.SIM_THRESHOLD).sum())} by cosine,"
        f" {int(pairs.contained.sum())} by containment), {len(sizes)} groups, of which"
        f" {int((sizes > 1).sum())} have more than one row ({int(sizes[sizes > 1].sum())} rows)."
        f" Largest group: {int(sizes.max())} rows.",
        "",
        "| group size | " + " | ".join(str(k) for k in size_counts.index) + " |",
        "|---" * (len(size_counts) + 1) + "|",
        "| groups | " + " | ".join(str(v) for v in size_counts.values) + " |",
        "",
        "Groups with more than one label value are kept whole; their rows keep their own labels.",
        f" Groups with mixed aspect: {int((full.groupby('group_id').aspect.nunique() > 1).sum())};"
        f" mixed sentiment: {int((full.groupby('group_id').sentiment.nunique() > 1).sum())}.",
        "",
        "## 2. Partitions and CV folds",
        "",
        f"`StratifiedGroupKFold(n_splits={sp.N_SPLITS}, shuffle=True, random_state={sp.SPLIT_SEED})` on"
        f" `aspect|sentiment` with `group_id`; fold {sp.TEST_FOLD} is the held-out **test** partition."
        f" Inside dev, the same splitter with seeds {', '.join(map(str, sp.CV_SEEDS))} gives"
        f" {len(sp.CV_SEEDS)} repeats of {sp.N_SPLITS}-fold CV (`-1` for test rows).",
        "",
        f"- dev: **{int((full.partition == 'dev').sum())}** rows · test: **{int((full.partition == 'test').sum())}**"
        f" rows ({(full.partition == 'test').mean() * 100:.1f}%)",
        "",
        "| task | label | dev % | test % | diff (pp) |",
        "|---|---|---|---|---|",
    ]
    for r in props.itertuples():
        out.append(f"| {r.task} | {r.label} | {r.dev_pct:.2f} | {r.test_pct:.2f} | {r.diff_pp:.2f} |")
    out += ["", f"Largest difference: {props.diff_pp.max():.2f} pp (acceptance: ≤ 2 pp).", ""]
    dev = full[full.partition == "dev"]
    joint = dev.aspect + "|" + dev.sentiment
    out += ["| repeat | fold sizes | smallest joint cell per fold |", "|---|---|---|"]
    for col in sp.CV_COLUMNS:
        fold_sizes = dev.groupby(col).size()
        smallest = pd.crosstab(dev[col], joint).min(axis=1)
        out.append(f"| `{col}` | {' / '.join(map(str, fold_sizes))} | {' / '.join(map(str, smallest))} |")
    out += ["", "## 3. Leakage checks (all must be 0)", "", "| check | count |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in checks.items()]
    out += [
        "",
        "The test partition is written to a local file but is not read by any analysis or experiment"
        " code before the one-shot final evaluation (milestone M5); experiments use `split.load_dev()`.",
    ]
    return "\n".join(out) + "\n"
