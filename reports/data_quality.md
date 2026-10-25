# Data quality report

Written by `python -m arabic_reviews validate` (deterministic; no timestamps).

- Source: `cleaned01_fixed.xlsx` — DOI `10.5281/zenodo.17781916` — SHA-256 `f171ef057abcae5946106534c8ef26cc9ef32a9d94fb8ec3e1843898f8f6d901`
- Output: `data/processed/validated.csv` (local only) — SHA-256 `66b938be391f8755fff4c638226eeb5fc09475863301b67c2d7faa62357b05d5`
- Rows: raw **3000** → validated **2769**

## Lineage (repair and exclusion steps)

| # | step | reason | before | modified | excluded | after |
|---|---|---|---|---|---|---|
| 1 | `strip_label_suffix` | label strings glued to the end of the text (CSV-merge artefact) | 3000 | 511 | 0 | 3000 |
| 2 | `strip_translation` | platform translation marker and Latin-only leading/trailing paragraphs | 3000 | 122 | 0 | 3000 |
| 3 | `clean_whitespace` | invisible/control characters, repeated spaces, blank lines | 3000 | 962 | 0 | 3000 |
| 4 | `exclude_no_arabic` | empty or no Arabic letters after repair | 3000 | 0 | 0 | 3000 |
| 5 | `deduplicate` | one representative per duplicate group; labels by per-task majority | 3000 | 2 | 217 | 2783 |
| 6 | `deduplicate_ties` | duplicate groups whose labels tie in a task | 2783 | 0 | 14 | 2769 |

Reconciliation: 3000 raw = 2769 kept + 0 no_arabic_text + 217 duplicate_copy + 14 duplicate_group_label_tie.

## Quality checks (rows affected)

| check | raw | validated |
|---|---|---|
| embedded_label_suffix | 511 | 0 |
| delimiter_label_scan | 511 | 0 |
| translation_marker | 9 | 0 |
| latin_letters | 145 | 24 |
| no_arabic_letters | 0 | 0 |
| invisible_characters | 11 | 0 |
| control_characters | 0 | 0 |
| diacritics | 187 | 171 |
| tatweel | 10 | 10 |
| letter_runs_3plus | 103 | 96 |
| emoji_or_symbols | 175 | 165 |
| digits | 109 | 98 |
| negation_particles | 500 | 458 |
| spreadsheet_cell_reference | 1 | 1 |
| exact_duplicates (rows in groups / groups) | 334 / 142 | 0 / 0 |
| normalised_duplicates (rows in groups / groups) | 369 / 155 | 0 / 0 |
| text length chars (raw): min / median / p95 / max | 4 / 35 / 128 / 845 | |
| text length chars (validated): min / median / p95 / max | 4 / 31 / 121 / 357 | |

`negation_particles`, `emoji_or_symbols`, `digits`, `diacritics` and `letter_runs_3plus` are
informative signals and are deliberately **not** repaired here; model-side handling is decided
by ablation inside the training pipeline.

## Label integrity

- Embedded label strings disagreeing with the official label: **26** (4 aspect, 22 sentiment). Logged as label noise; the official labels are kept.
- Duplicate groups: **162**; excluded for a label tie: **7**; labels changed by per-task majority: **2**.
- Every other kept record carries its source label unchanged; labels, group sizes and the
  kept/excluded partition are recomputed from the source on every run.
- Aspect-only suffixes removed (`:aspect`, audited in `lineage.json`): **4**.
- Translation paragraphs removed that contained pasted Arabic aspect words (audited): **4**.
- Residual label strings after repair (end-anchored pattern / delimiter scan anywhere): **0 / 0**.
- Known residual noise left as is: Latin words inside Arabic sentences or bilingual lines written by the reviewer, and one spreadsheet cell reference inside a text.

## Label distribution — raw

| aspect \ sentiment | ايجابي | سلبي | محايد | total |
|---|---|---|---|---|
| المأكولات | 324 | 63 | 52 | 439 |
| المشروبات | 726 | 137 | 64 | 927 |
| الخدمة | 298 | 330 | 74 | 702 |
| الجلسات | 370 | 106 | 46 | 522 |
| الاسعار | 144 | 226 | 40 | 410 |
| total | 1862 | 862 | 276 | 3000 |

## Label distribution — validated

| aspect \ sentiment | ايجابي | سلبي | محايد | total |
|---|---|---|---|---|
| المأكولات | 313 | 58 | 48 | 419 |
| المشروبات | 685 | 132 | 62 | 879 |
| الخدمة | 270 | 299 | 70 | 639 |
| الجلسات | 364 | 104 | 45 | 513 |
| الاسعار | 139 | 150 | 30 | 319 |
| total | 1771 | 743 | 255 | 2769 |
