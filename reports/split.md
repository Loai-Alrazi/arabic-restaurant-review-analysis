# Split report

Written by `python -m arabic_reviews split` (deterministic; no timestamps).

- Input: `data/processed/validated.csv` — SHA-256 `66b938be391f8755fff4c638226eeb5fc09475863301b67c2d7faa62357b05d5` — 2769 rows
- Output: `data/splits/split_v1.csv` (published, no text: `rid, group_id, partition, cv_fold_r0, cv_fold_r1, cv_fold_r2`), `reports/near_duplicates.csv` (pairs by rid); `data/processed/dev.csv` and `test.csv` stay local.

## 1. Near-duplicate grouping (label-free)

Keys: `text.dedup_key`. Similarity: cosine on character 3–5-gram TF-IDF (`char_wb`, sublinear tf), fitted on all validated texts — a quality step that never sees labels. A pair is linked when cosine ≥ **0.80**, or when one key of at least **4 words** occurs inside another. Groups are connected components (union-find); `group_id` is the smallest `rid` in the group.

Pairs per similarity band (all row pairs, before the containment rule):

| cosine | >=0.60 | >=0.70 | >=0.75 | >=0.80 | >=0.85 | >=0.90 | >=0.95 |
|---|---|---|---|---|---|---|---|
| pairs | 584 | 246 | 155 | 96 | 59 | 33 | 14 |

Threshold choice (from reading sampled pairs in every band from 0.60 to 1.00):

- **≥ 0.80**: the same review copied with a few words cut or added (for example rids 681/785, 436/1336, 446/1346, 569/607) or the same short stock phrase ("quiet place" / "quiet place, very"). Linking stock phrases is conservative: a model cannot score by recognising a phrase it saw in training.
- **0.70–0.80**: mostly different reviews sharing vocabulary, some with different meaning or label (for example rids 248/249 differ only in *coffee* vs *sweets*), mixed with a few truncated copies (rid 364 is the tail of rids 407 and 1311). Grouping by cosine here would merge distinct reviews.
- The containment rule recovers truncated copies whose cosine falls below the threshold (20 pairs, for example rids 344/375, 438/1338, 1795/2552). It requires 4+ words; shorter keys ("good coffee") are common phrases rather than copies.

Result: 116 linked pairs (96 by cosine, 40 by containment), 2658 groups, of which 86 have more than one row (197 rows). Largest group: 6 rows.

| group size | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| groups | 2572 | 68 | 13 | 4 | 1 |

Groups with more than one label value are kept whole; their rows keep their own labels.
 Groups with mixed aspect: 9; mixed sentiment: 15.

## 2. Partitions and CV folds

`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on `aspect|sentiment` with `group_id`; fold 0 is the held-out **test** partition. Inside dev, the same splitter with seeds 42, 43, 44 gives 3 repeats of 5-fold CV (`-1` for test rows).

- dev: **2215** rows · test: **554** rows (20.0%)

| task | label | dev % | test % | diff (pp) |
|---|---|---|---|---|
| aspect | المأكولات | 15.17 | 14.98 | 0.19 |
| aspect | المشروبات | 31.74 | 31.77 | 0.03 |
| aspect | الخدمة | 23.07 | 23.10 | 0.03 |
| aspect | الجلسات | 18.51 | 18.59 | 0.08 |
| aspect | الاسعار | 11.51 | 11.55 | 0.04 |
| sentiment | ايجابي | 63.97 | 63.90 | 0.07 |
| sentiment | سلبي | 26.86 | 26.71 | 0.15 |
| sentiment | محايد | 9.16 | 9.39 | 0.22 |

Largest difference: 0.22 pp (acceptance: ≤ 2 pp).

| repeat | fold sizes | smallest joint cell per fold |
|---|---|---|
| `cv_fold_r0` | 444 / 443 / 443 / 443 / 442 | 5 / 5 / 5 / 4 / 5 |
| `cv_fold_r1` | 443 / 443 / 443 / 443 / 443 | 4 / 5 / 5 / 5 / 5 |
| `cv_fold_r2` | 444 / 443 / 442 / 443 / 443 | 5 / 5 / 4 / 5 / 5 |

## 3. Leakage checks (all must be 0)

| check | count |
|---|---|
| partition: groups in >1 part | 0 |
| partition: keys in >1 part | 0 |
| partition: near-duplicate pairs across parts | 0 |
| cv_fold_r0: groups in >1 part | 0 |
| cv_fold_r0: keys in >1 part | 0 |
| cv_fold_r0: near-duplicate pairs across parts | 0 |
| cv_fold_r1: groups in >1 part | 0 |
| cv_fold_r1: keys in >1 part | 0 |
| cv_fold_r1: near-duplicate pairs across parts | 0 |
| cv_fold_r2: groups in >1 part | 0 |
| cv_fold_r2: keys in >1 part | 0 |
| cv_fold_r2: near-duplicate pairs across parts | 0 |
| test rows with a CV fold | 0 |
| dev rows without a CV fold | 0 |

The test partition is written to a local file but is not read by any analysis or experiment code before the one-shot final evaluation (milestone M5); experiments use `split.load_dev()`.
