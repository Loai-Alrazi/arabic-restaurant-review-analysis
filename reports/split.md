# Split report

Written by `python -m arabic_reviews split` (deterministic; no timestamps).

- Input: `data/processed/validated.csv` — SHA-256 `66b938be391f8755fff4c638226eeb5fc09475863301b67c2d7faa62357b05d5` — 2769 rows
- Output: `data/splits/split_v1.csv` (published, no text: `rid, group_id, partition, cv_fold_r0, cv_fold_r1, cv_fold_r2`), `reports/near_duplicates.csv` (pairs by rid); `data/processed/dev.csv` and `test.csv` stay local.

## 1. Near-duplicate grouping (label-free)

Keys: `text.dedup_key`. The linking rules never read labels. Two reviews are linked when any rule fires:

1. **cosine** ≥ 0.80 on character 3–5-gram TF-IDF (`char_wb`, sublinear tf) fitted on all validated texts (a quality step; it feeds nothing in model selection);
2. **form**: identical keys once spaces and a leading و/ف/ب and article ال on each word are removed;
3. **edit**: character edit ratio ≥ 0.9 on space-free keys (pairs with cosine ≥ 0.5);
4. **chars**: ≥ 90% of the shorter key's character 4-grams occur in the other key;
5. **words**: ≥ 80% of the shorter key's words matched in order by runs of ≥ 2 words, or a common run of ≥ 5 consecutive words.

Rules 3–5 apply only when the shorter key has ≥ 4 words. Groups are connected components (union-find); `group_id` is the smallest `rid` in the group.

Pairs per cosine band (all row pairs):

| cosine | >=0.60 | >=0.70 | >=0.75 | >=0.80 | >=0.85 | >=0.90 | >=0.95 |
|---|---|---|---|---|---|---|---|
| pairs | 584 | 246 | 155 | 96 | 59 | 33 | 14 |

How the rules were chosen: sampled pairs in every band from 0.60 to 1.00 were read together with their labels; the linking algorithm itself is label-free.

- **≥ 0.80**: the same review copied with a few words cut or added (for example rids 681/785, 436/1336, 446/1346, 569/607), or the same short stock phrase. Linking stock phrases is conservative.
- **Below 0.80** cosine mixes reviews that only share vocabulary with copies that symmetric cosine misses: fragments of a longer review (rid 364 is the tail of 407 and 1311), copies with glued words or different line breaks, and re-spaced or one-word-edited copies. Rules 2–5 link these. A recurring source pattern is the same review with the aspect word swapped (248/249 coffee/sweets, 407/1311, 759/1658); these are linked, since for sentiment they are the same text.
- **Short phrases**: 2–3-word phrases that differ only in clitics or spacing ('good coffee' vs 'the good coffee') have the same bare key and are linked by the `form` rule. Phrases that differ in their content word are not linked (rids 149/1863, 'quiet and beautiful place' vs 'quiet and tidy place'); new reviews contain such phrases too, so they are part of the task rather than copies.
- **Template families are only partly linked**: the `words` rule needs 80% of the shorter text matched, so a 5-word template with one word swapped is linked while a 4-word one is not. A few such pairs may fall into different CV folds (a small optimism in development scores) or across the partition boundary.
- **Over-linking**: a shared closing phrase or a stock phrase can join two otherwise unrelated reviews; this keeps both in one partition and costs only a little stratification freedom.
- **Limits of the leakage checks**: the counts below test the linking itself and are true by construction; the independent test in `tests/test_split.py` uses measures that are special cases of the rules. Residual copies that no rule links cannot be excluded without reading the held-out texts, which is not done.

Linked pairs by rule (cosine first, otherwise the first copy rule that fired): chars 26, cosine 96, edit 4, form 10, words 40.

Result: 176 linked pairs, 2611 groups, of which 112 have more than one row (270 rows). Largest group: 9 rows.

| group size | 1 | 2 | 3 | 4 | 5 | 6 | 9 |
|---|---|---|---|---|---|---|---|
| groups | 2499 | 87 | 12 | 10 | 1 | 1 | 1 |

Groups with more than one label value are kept whole; their rows keep their own labels.
 Groups with mixed aspect: 15; mixed sentiment: 19.

## 2. Partitions and CV folds

`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on `aspect|sentiment` with `group_id`; fold 0 is the held-out **test** partition. Inside dev, the same splitter with seeds 42, 43, 44 gives 3 repeats of 5-fold CV (`-1` for test rows).

- dev: **2215** rows · test: **554** rows (20.0%)

| task | label | dev % | test % | diff (pp) |
|---|---|---|---|---|
| aspect | المأكولات | 15.17 | 14.98 | 0.19 |
| aspect | المشروبات | 31.69 | 31.95 | 0.26 |
| aspect | الخدمة | 23.12 | 22.92 | 0.19 |
| aspect | الجلسات | 18.51 | 18.59 | 0.08 |
| aspect | الاسعار | 11.51 | 11.55 | 0.04 |
| sentiment | ايجابي | 63.97 | 63.90 | 0.07 |
| sentiment | سلبي | 26.86 | 26.71 | 0.15 |
| sentiment | محايد | 9.16 | 9.39 | 0.22 |

Largest difference: 0.26 pp (acceptance: ≤ 2 pp).

| repeat | fold sizes | smallest joint cell per fold |
|---|---|---|
| `cv_fold_r0` | 444 / 443 / 443 / 442 / 443 | 5 / 5 / 5 / 4 / 5 |
| `cv_fold_r1` | 443 / 444 / 443 / 442 / 443 | 5 / 4 / 5 / 5 / 5 |
| `cv_fold_r2` | 444 / 443 / 443 / 443 / 442 | 5 / 5 / 5 / 5 / 4 |

## 3. Leakage checks (all must be 0)

| check | count |
|---|---|
| partition: groups in >1 part | 0 |
| partition: bare keys in >1 part | 0 |
| partition: near-duplicate pairs across parts | 0 |
| cv_fold_r0: groups in >1 part | 0 |
| cv_fold_r0: bare keys in >1 part | 0 |
| cv_fold_r0: near-duplicate pairs across parts | 0 |
| cv_fold_r1: groups in >1 part | 0 |
| cv_fold_r1: bare keys in >1 part | 0 |
| cv_fold_r1: near-duplicate pairs across parts | 0 |
| cv_fold_r2: groups in >1 part | 0 |
| cv_fold_r2: bare keys in >1 part | 0 |
| cv_fold_r2: near-duplicate pairs across parts | 0 |
| test rows with a CV fold | 0 |
| dev rows without a CV fold | 0 |

The test partition is written to a local file but is not read by any analysis or experiment code before the one-shot final evaluation (milestone M5); experiments use `split.load_dev()`.
