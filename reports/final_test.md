# Final test (held-out partition, read once)

- Run 1, 2026-10-25T10:29:29+00:00, commit `b577bb0`
- Protocol fingerprint `ec20dc6ba81703194ac2b536f51698cb75d38427d44fa04f6d2f6401a7977b41` (equal to `reports/protocol.json`, checked before the test file was opened)
- Test partition: 554 reviews, SHA-256 `142bac8eafa549b39558c8a27c429d8cf936e4f4624521f56dbc7856f6c3c627`; models trained on all 2215 development reviews
- Intervals: 95% percentile bootstrap, 2000 resamples (seed 0), near-duplicate groups as the resampling unit
- Classes and label cells with fewer than 20 test rows (a limit fixed before the test was read) are marked and not used for any claim

## sentiment

| model | Macro-F1 | 95% interval | balanced accuracy | accuracy |
|---|---|---|---|---|
| final `m4_logreg_C0.3` | 0.6564 | [0.604, 0.704] | 0.6704 | 0.7635 |
| reference `logreg_word11` | 0.5547 | [0.507, 0.605] | 0.5502 | 0.7870 |
| `dummy_most_frequent` | 0.2599 | [0.249, 0.270] | 0.3333 | 0.6390 |
| `dummy_stratified` | 0.3052 | [0.276, 0.336] | 0.3078 | 0.4892 |

| paired difference (final minus) | observed | 95% interval | share of resamples <= 0 |
|---|---|---|---|
| reference | +0.1017 | [0.044, 0.162] | 0.001 |
| dummy_most_frequent | +0.3965 | [0.342, 0.447] | 0.000 |
| dummy_stratified | +0.3512 | [0.290, 0.411] | 0.000 |

Decision (frozen rule): beats the dummies (dummy_most_frequent: yes, dummy_stratified: yes); versus the reference: above; success: yes.

Expected range from the fresh CV seeds: final [0.607, 0.706] -> test 0.6564 is inside; reference [0.477, 0.586] -> 0.5547 is inside.

Per class (final model):

| class | precision | recall | F1 | 95% interval of F1 | test rows |
|---|---|---|---|---|---|
| ايجابي | 0.873 | 0.797 | 0.833 | [0.799, 0.863] | 354 |
| سلبي | 0.674 | 0.811 | 0.736 | [0.681, 0.788] | 148 |
| محايد | 0.396 | 0.404 | 0.400 | [0.274, 0.517] | 52 |

Confusion matrix (final model; rows = true, columns = predicted):

| true \ predicted | ايجابي | سلبي | محايد |
|---|---|---|---|
| ايجابي | 282 | 46 | 26 |
| سلبي | 22 | 120 | 6 |
| محايد | 19 | 12 | 21 |

Most frequent confusions: ايجابي -> سلبي: 46; ايجابي -> محايد: 26; سلبي -> ايجابي: 22 (94 of 131 errors).

Certainty levels on the test partition (thresholds frozen in the protocol):

| level | test rows | share | accuracy | fresh-seed OOF share | OOF accuracy |
|---|---|---|---|---|---|
| high | 202 | 0.365 | 0.946 | 0.352 | 0.937 |
| medium | 156 | 0.282 | 0.756 | 0.270 | 0.817 |
| low | 196 | 0.354 | 0.582 | 0.378 | 0.593 |

## aspect

| model | Macro-F1 | 95% interval | balanced accuracy | accuracy |
|---|---|---|---|---|
| final `m4_logreg_C1` | 0.9009 | [0.873, 0.925] | 0.8967 | 0.9007 |
| reference `logreg_word11` | 0.8252 | [0.789, 0.858] | 0.8058 | 0.8321 |
| `dummy_most_frequent` | 0.0969 | [0.086, 0.107] | 0.2000 | 0.3195 |
| `dummy_stratified` | 0.2131 | [0.179, 0.247] | 0.2138 | 0.2329 |

| paired difference (final minus) | observed | 95% interval | share of resamples <= 0 |
|---|---|---|---|
| reference | +0.0757 | [0.047, 0.105] | 0.000 |
| dummy_most_frequent | +0.8041 | [0.775, 0.830] | 0.000 |
| dummy_stratified | +0.6878 | [0.647, 0.727] | 0.000 |

Decision (frozen rule): beats the dummies (dummy_most_frequent: yes, dummy_stratified: yes); versus the reference: above; success: yes.

Expected range from the fresh CV seeds: final [0.879, 0.936] -> test 0.9009 is inside; reference [0.759, 0.831] -> 0.8252 is inside.

Per class (final model):

| class | precision | recall | F1 | 95% interval of F1 | test rows |
|---|---|---|---|---|---|
| المأكولات | 0.960 | 0.867 | 0.911 | [0.859, 0.952] | 83 |
| المشروبات | 0.954 | 0.932 | 0.943 | [0.915, 0.966] | 177 |
| الخدمة | 0.814 | 0.898 | 0.854 | [0.808, 0.895] | 127 |
| الجلسات | 0.856 | 0.864 | 0.860 | [0.804, 0.906] | 103 |
| الاسعار | 0.952 | 0.922 | 0.937 | [0.883, 0.976] | 64 |

Confusion matrix (final model; rows = true, columns = predicted):

| true \ predicted | المأكولات | المشروبات | الخدمة | الجلسات | الاسعار |
|---|---|---|---|---|---|
| المأكولات | 72 | 5 | 5 | 1 | 0 |
| المشروبات | 2 | 165 | 5 | 3 | 2 |
| الخدمة | 1 | 0 | 114 | 11 | 1 |
| الجلسات | 0 | 1 | 13 | 89 | 0 |
| الاسعار | 0 | 2 | 3 | 0 | 59 |

Most frequent confusions: الجلسات -> الخدمة: 13; الخدمة -> الجلسات: 11; المشروبات -> الخدمة: 5 (29 of 55 errors).

Certainty levels on the test partition (thresholds frozen in the protocol):

| level | test rows | share | accuracy | fresh-seed OOF share | OOF accuracy |
|---|---|---|---|---|---|
| high | 408 | 0.736 | 0.978 | 0.713 | 0.981 |
| medium | 84 | 0.152 | 0.786 | 0.181 | 0.812 |
| low | 62 | 0.112 | 0.548 | 0.106 | 0.548 |

## Both labels

Joint accuracy (both labels right, final models): 0.6986; expected range [0.665, 0.746] -> inside.

| aspect | sentiment | test rows | joint accuracy |
|---|---|---|---|
| الاسعار | ايجابي | 28 | 0.536 |
| الاسعار | سلبي | 30 | 0.833 |
| الاسعار | محايد | 6 | 0.333 (too small for a claim) |
| الجلسات | ايجابي | 73 | 0.795 |
| الجلسات | سلبي | 21 | 0.381 |
| الجلسات | محايد | 9 | 0.667 (too small for a claim) |
| الخدمة | ايجابي | 54 | 0.574 |
| الخدمة | سلبي | 59 | 0.898 |
| الخدمة | محايد | 14 | 0.286 (too small for a claim) |
| المأكولات | ايجابي | 62 | 0.823 |
| المأكولات | سلبي | 11 | 0.364 (too small for a claim) |
| المأكولات | محايد | 10 | 0.300 (too small for a claim) |
| المشروبات | ايجابي | 137 | 0.803 |
| المشروبات | سلبي | 27 | 0.519 |
| المشروبات | محايد | 13 | 0.231 (too small for a claim) |

These numbers are reported as found. No model, threshold or feature was changed after the test partition was read; a second reading of the test partition needs a written methodological reason and is logged in `reports/test_access_log.json`.
