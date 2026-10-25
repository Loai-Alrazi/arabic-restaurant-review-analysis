# Version 2 - sentiment on development data only

Plan and decision rules: `docs/V2_PLAN.md`. Every number below comes from the development partition (2215 reviews, `dev.csv` SHA-256 `f0d6dc75ebd5...`); the old held-out test was not opened by this run (files opened under data/, reports/, models/: `data/processed/dev.csv`, `data/splits/split_v1.csv`, `reports/experiments.csv`, `reports/protocol.json`, `reports/v2/blind_reading.csv`, `reports/v2/experiments.csv`). Polarity-cue file SHA-256 `ac5460bb887a...` (frozen before any candidate was scored). Commit `0b262c8`.

**What these numbers can and cannot show.** The design of version 2 (its target, candidates and cue list) came after reading development reviews, and the same reviews are used here. The nested CV estimates the whole selection procedure without fold-selection optimism, and the outer folds use new seeds, but no new reviews: the estimates are development-only and optimistic to an unknown degree. There is no independent test for version 2, and no result is shown on the old test.

## 1. Diagnosis of the neutral class

Class counts: ايجابي 1417, سلبي 595, محايد 203.

Out-of-fold confusion of V1 on the selection folds (3 repeats pooled, every review three times; rows = true, columns = predicted):

| true \ pred | ايجابي | سلبي | محايد |
|---|---|---|---|
| ايجابي | 3538 | 465 | 248 |
| سلبي | 241 | 1357 | 187 |
| محايد | 159 | 203 | 247 |

Share of reviews with each cue type (fixed list, `src/arabic_reviews/cues.py`):

| cue | ايجابي | سلبي | محايد |
|---|---|---|---|
| positive | 75.9% | 15.0% | 55.7% |
| negative | 3.4% | 53.8% | 33.0% |
| lukewarm | 1.6% | 5.7% | 18.2% |
| contrast | 3.8% | 19.5% | 44.8% |
| negated_positive | 0.6% | 6.7% | 4.9% |
| mixed | 3.1% | 7.1% | 23.2% |
| positive_and_contrast | 3.0% | 6.7% | 34.5% |
| no_cue | 22.3% | 35.0% | 20.7% |
| any | 77.7% | 65.0% | 79.3% |

Sentiment agreement inside near-duplicate groups (pairs of rows of one group; the texts differ, for example a full review and a fragment of it):

| pairs involving | pairs | agreement |
|---|---|---|
| all | 177 | 0.904 |
| ايجابي | 109 | 0.927 |
| سلبي | 68 | 0.809 |
| محايد | 17 | 0.235 |

Blind reading of a shuffled sample of 120 development reviews (60 neutral, 30 positive, 30 negative), done without the labels in a context separate from this analysis; one reading, not a second human annotation and not a measure of annotator agreement. Categories per rid: `reports/v2/blind_reading.csv`.

| gold \ reading | positive | negative | mixed | lukewarm | descriptive | unclear |
|---|---|---|---|---|---|---|
| ايجابي | 26 | 2 | 1 | 0 | 1 | 0 |
| سلبي | 0 | 26 | 1 | 0 | 3 | 0 |
| محايد | 12 | 13 | 29 | 4 | 2 | 0 |

## 2. Selection (seeds 42-44, 15 paired folds)

| candidate | Macro-F1 (mean ± sd) | F1 ايجابي | F1 سلبي | F1 محايد | gain vs V1 | folds won | neutral F1 change | eligible |
|---|---|---|---|---|---|---|---|---|
| V1 | 0.6529 ± 0.0244 | 0.864 | 0.712 | 0.382 | – | – | – | reference |
| H1_phrases | 0.6560 ± 0.0235 | 0.866 | 0.715 | 0.387 | +0.0031 | 9/15 | +0.0047 | no |
| H2_cues | 0.6781 ± 0.0180 | 0.877 | 0.708 | 0.449 | +0.0252 | 11/15 | +0.0667 | yes |
| H3_both | 0.6785 ± 0.0200 | 0.877 | 0.711 | 0.447 | +0.0256 | 12/15 | +0.0650 | yes |

Best single candidate by Macro-F1: `H2_cues`; H3 against it: +0.0004, 8/15 folds (does not pass). **Chosen by the rule: `H2_cues`.** The rule is a deterministic selection heuristic, not a significance test: the 15 folds are correlated (each review appears in three repeats).

Ablation (2 x 2, same folds): V1 = neither change, H1 = word bigrams, H2 = cue block, H3 = both; the rows above are the four cells.

## 3. Nested CV estimate of the selection procedure (outer seeds 300-302)

Inside each of the 15 outer folds the same rule chose a candidate on a grouped inner 5-fold CV of the outer-training rows (fold wins >= 4 of 5), and the chosen candidate predicted the outer fold. Choices: V1 3, H1_phrases 0, H2_cues 12, H3_both 0.

| comparison on the outer folds | Macro-F1 gain [95% interval] | ايجابي F1 change | سلبي F1 change | محايد F1 change | folds won |
|---|---|---|---|---|---|
| procedure vs V1 | +0.0178 [-0.001, +0.036] | +0.0106 [+0.001, +0.020] | +0.0030 [-0.016, +0.023] | +0.0397 [-0.003, +0.080] | 10/15 |
| H2_cues vs V1 (stability) | +0.0287 [+0.005, +0.051] | +0.0133 [+0.002, +0.025] | +0.0054 [-0.019, +0.030] | +0.0672 [+0.012, +0.119] | 13/15 |

Intervals: paired group bootstrap, 2000 resamples (seed 0); each resample draws the near-duplicate groups once, applies them to the three repeats, computes the metric within each repeat and averages the three differences.

**Decision (plan section 5): no new sentiment model adopted; version 2 keeps the version-1 sentiment configuration.**

Outer-fold scores of every candidate (no choice made on them):

| candidate | Macro-F1 (mean ± sd) | F1 ايجابي | F1 سلبي | F1 محايد |
|---|---|---|---|---|
| V1 | 0.6502 ± 0.0209 | 0.863 | 0.710 | 0.378 |
| H1_phrases | 0.6500 ± 0.0254 | 0.865 | 0.714 | 0.371 |
| H2_cues | 0.6793 ± 0.0206 | 0.876 | 0.715 | 0.447 |
| H3_both | 0.6794 ± 0.0203 | 0.876 | 0.714 | 0.448 |
| procedure (nested) | 0.6681 ± 0.0294 | 0.874 | 0.713 | 0.418 |

Joint accuracy (sentiment and aspect both right; aspect = the version-1 model): V1: selection 0.7060, outer 0.7047.

## 4. Certainty levels per predicted class

Score = the top class probability of the released model. Per predicted class, from the pooled out-of-fold predictions of the selection folds (support counted in distinct near-duplicate groups): high starts at the smallest value t on the grid 0.30, 0.35, ..., 0.95 whose tail (score >= t) is right >= 90% of the time over >= 30 groups and whose lower edge [t, t + 0.10) is right >= 85% of the time when it holds >= 30 groups; medium likewise for [t, high) with 70% / 65%. A dash means the level does not exist for that class. A level whose accuracy on the outer folds falls more than 0.05 below its target is merged into the next lower one.

**Released rule (plan section 9, decided after the registered rule's results were seen):** the same search, but no level may start below its version-1 threshold, so a level can only be raised or removed for a class, never widened. The registered rule widened high for the largest classes (for example positive from 0.60 to 0.45), which in turn showed high certainty on hand-written descriptive sentences (section 6); the aim was to stop over-claiming, so the release keeps the stricter variant. Both are reported.

### sentiment

Version-1 thresholds (one pair for every class): high 0.60, medium 0.50.

| predicted | released: high from | released: medium from | registered: high from | registered: medium from |
|---|---|---|---|---|
| ايجابي | 0.60 | 0.50 | 0.45 | 0.35 |
| سلبي | 0.70 | 0.50 | 0.70 | 0.50 |
| محايد | – | – | – | – |

Dropped by the outer-fold check: released none; registered none.

| predicted | level | released: selection rows (groups) | accuracy | released: outer rows (groups) | accuracy | v1 rule: outer rows | accuracy | registered: outer rows | accuracy |
|---|---|---|---|---|---|---|---|---|---|
| ايجابي | high | 1742 (631) | 0.978 | 1785 (643) | 0.977 | 1785 | 0.977 | 3329 | 0.936 |
| ايجابي | medium | 1066 (519) | 0.922 | 1026 (507) | 0.918 | 1026 | 0.918 | 600 | 0.690 |
| ايجابي | low | 1130 (529) | 0.754 | 1122 (516) | 0.754 | 1122 | 0.754 | 4 | 0.250 |
| سلبي | high | 136 (57) | 0.926 | 143 (60) | 0.902 | 455 | 0.864 | 143 | 0.902 |
| سلبي | medium | 923 (376) | 0.794 | 925 (379) | 0.786 | 613 | 0.755 | 925 | 0.786 |
| سلبي | low | 966 (460) | 0.516 | 1003 (480) | 0.510 | 1003 | 0.510 | 1003 | 0.510 |
| محايد | high | 0 (0) | – | 0 (0) | – | 107 | 0.626 | 0 | – |
| محايد | medium | 0 (0) | – | 0 (0) | – | 142 | 0.415 | 0 | – |
| محايد | low | 682 (296) | 0.362 | 641 (283) | 0.370 | 392 | 0.283 | 641 | 0.370 |

### aspect

Version-1 thresholds (one pair for every class): high 0.50, medium 0.35.

| predicted | released: high from | released: medium from | registered: high from | registered: medium from |
|---|---|---|---|---|
| المأكولات | 0.50 | 0.35 | 0.35 | – |
| المشروبات | 0.50 | 0.35 | 0.35 | 0.30 |
| الخدمة | 0.50 | 0.35 | 0.45 | 0.40 |
| الجلسات | 0.50 | 0.35 | 0.45 | 0.35 |
| الاسعار | 0.50 | – | 0.30 | – |

Dropped by the outer-fold check: released none; registered none.

| predicted | level | released: selection rows (groups) | accuracy | released: outer rows (groups) | accuracy | v1 rule: outer rows | accuracy | registered: outer rows | accuracy |
|---|---|---|---|---|---|---|---|---|---|
| المأكولات | high | 727 (243) | 0.981 | 723 (244) | 0.976 | 723 | 0.976 | 899 | 0.958 |
| المأكولات | medium | 173 (76) | 0.884 | 176 (91) | 0.881 | 176 | 0.881 | 0 | – |
| المأكولات | low | 91 (52) | 0.484 | 88 (50) | 0.580 | 88 | 0.580 | 88 | 0.580 |
| المشروبات | high | 1523 (498) | 0.986 | 1530 (501) | 0.986 | 1530 | 0.986 | 1841 | 0.968 |
| المشروبات | medium | 313 (140) | 0.895 | 311 (140) | 0.881 | 311 | 0.881 | 124 | 0.758 |
| المشروبات | low | 202 (104) | 0.663 | 204 (101) | 0.672 | 204 | 0.672 | 80 | 0.537 |
| الخدمة | high | 1019 (350) | 0.972 | 1025 (350) | 0.972 | 1025 | 0.972 | 1150 | 0.955 |
| الخدمة | medium | 393 (182) | 0.715 | 399 (187) | 0.689 | 399 | 0.689 | 114 | 0.719 |
| الخدمة | low | 287 (142) | 0.456 | 271 (136) | 0.480 | 271 | 0.480 | 431 | 0.513 |
| الجلسات | high | 840 (278) | 0.979 | 839 (277) | 0.977 | 839 | 0.977 | 930 | 0.966 |
| الجلسات | medium | 258 (116) | 0.791 | 254 (113) | 0.799 | 254 | 0.799 | 163 | 0.767 |
| الجلسات | low | 102 (54) | 0.559 | 108 (59) | 0.528 | 108 | 0.528 | 108 | 0.528 |
| الاسعار | high | 611 (180) | 0.985 | 611 (179) | 0.985 | 611 | 0.985 | 693 | 0.978 |
| الاسعار | medium | 0 (0) | – | 0 (0) | – | 54 | 0.963 | 0 | – |
| الاسعار | low | 106 (40) | 0.906 | 106 (48) | 0.887 | 52 | 0.808 | 24 | 0.750 |

## 5. Calibration (selection folds, pooled)

The interface never shows probabilities; this section explains why the levels are read from measured accuracy and not from the probability itself.

| model | top-label ECE | signed gap (mean top probability - accuracy) | multiclass Brier | mean top probability | accuracy |
|---|---|---|---|---|---|
| V1 | 0.214 | -0.214 | 0.400 | 0.560 | 0.774 |

## 6. Hand-written challenge sentences (qualitative only)

Written for this check, not taken from the data and without ground truth; models fitted on the whole development partition; the level by each rule.

| sentence | listed cue | V1, version-1 rule | V1, registered per-class rule | V1, released rule |
|---|---|---|---|---|
| الفرع في شارع التحلية | no | سلبي (medium) | سلبي (medium) | سلبي (medium) |
| يقدمون القهوة في أكواب ورقية | no | ايجابي (low) | ايجابي (high) | ايجابي (low) |
| زرت المقهى يوم الجمعة مع أخي | no | ايجابي (medium) | ايجابي (high) | ايجابي (medium) |
| القهوة لذيذة لكن الأسعار مرتفعة | yes | محايد (low) | محايد (low) | محايد (low) |
| ممتاز بس مزدحم | yes | محايد (high) | محايد (low) | محايد (low) |
| الخدمة لا بأس بها | yes | سلبي (low) | سلبي (low) | سلبي (low) |
| القهوة عادية | yes | محايد (low) | محايد (low) | محايد (low) |
| ما عجبني الكيك | yes | سلبي (low) | سلبي (low) | سلبي (low) |
| الخدمة ممتازة | yes | ايجابي (high) | ايجابي (high) | ايجابي (high) |
| الأسعار غالية جدا | yes | سلبي (high) | سلبي (medium) | سلبي (medium) |
