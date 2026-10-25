# Final evaluation protocol (frozen before the held-out test is read)

This file and `reports/protocol.json` fix every choice that the one-shot test in M5 depends on. All numbers come from the development partition only; the held-out test partition has not been read by any step up to here (experiment code can only call `split.load_dev()`).

- Protocol fingerprint (configurations + certainty thresholds): `ec20dc6ba81703194ac2b536f51698cb75d38427d44fa04f6d2f6401a7977b41`
- Development partition: 2215 reviews, `data/processed/dev.csv` SHA-256 `f0d6dc75ebd542d24cb04993bd4a492c13bd14433d91b05b58828acbd795c708`
- Split file `data/splits/split_v1.csv` SHA-256 `6a8a686fc143992313443060c5106ed80dd78a09013e85665bbeea39930a70f0`

## 1. Frozen configurations

| task | role | configuration | model | features | normalisation | class weight | parameters | selection CV Macro-F1 (3 x 5) |
|---|---|---|---|---|---|---|---|---|
| sentiment | final | `m4_logreg_C0.3` | logreg | word11+char25 TF-IDF | none | balanced | `{"C": 0.3}` | 0.6529 ± 0.0244 |
| sentiment | reference | `logreg_word11` | logreg | word11 TF-IDF | none | None | `{}` | 0.5236 ± 0.0285 |
| aspect | final | `m4_logreg_C1` | logreg | char25 TF-IDF | none | balanced | `{"C": 1}` | 0.9069 ± 0.0157 |
| aspect | reference | `logreg_word11` | logreg | word11 TF-IDF | none | None | `{}` | 0.7923 ± 0.0157 |

`word11` is word unigrams, `char25` is character 2-5 grams inside word boundaries, `+` joins both in one feature union. Every pipeline parameter, library defaults included, is listed under `pipelines` in `reports/protocol.json` and is part of the fingerprint. The dummy baselines (`most_frequent`, `stratified`) are fitted and reported next to the models in M5.

How the configurations were chosen (all on the same 3 x 5 grouped CV; rule: mean paired Macro-F1 gain >= 0.01 and a gain in >= 11 of 15 folds):

- Representation and preprocessing: `reports/ablations.md` (M3 final configuration per task: sentiment `a6_balanced`, aspect `a6_balanced`).
- sentiment, model family and regularisation (`reports/models.md`): families m4_linsvm -0.0217, 2/15 (rejected); m4_cnb -0.1707, 0/15 (rejected); family kept: logreg. One-standard-error pick on the C grid: 0.3; acceptance check against the default: not needed (not weaker-regularised than the default); value used: C = 0.3. Final vs the M3 configuration: -0.0031 mean, 6/15 folds, i.e. within the fold noise: the value comes from the procedure fixed before the grid was run, not from being better.
- aspect, model family and regularisation (`reports/models.md`): families m4_linsvm +0.0070, 11/15 (rejected); m4_cnb -0.0063, 5/15 (rejected); family kept: logreg. One-standard-error pick on the C grid: 3; acceptance check against the default: +0.0032, 10/15 folds, failed; value used: C = 1. Final vs the M3 configuration: +0.0000 mean, 0/15 folds, i.e. within the fold noise: the value comes from the procedure fixed before the grid was run, not from being better.
- The optional pre-trained sentence-vector experiment was not run: its pre-declared condition was not met (`reports/error_analysis.md`, per-review categories in `reports/error_categories.csv`).

## 2. Certainty levels

Score: the highest class probability of the task's logistic-regression pipeline. Thresholds come from out-of-fold predictions of the frozen configuration on the three selection repeats (seeds 42-44). Scores are cut into bands of width 0.05; a level starts at the smallest band edge from which every band with at least 30 out-of-fold rows is right at least as often as its target (high: 90%, medium: 70%); lower scores are low. The interface shows the level in words, never the raw probability.

| task | high from | medium from | level | selection OOF rows | share | accuracy | fresh-seed OOF share | accuracy |
|---|---|---|---|---|---|---|---|---|
| sentiment | 0.60 | 0.50 | high | 2295 | 0.345 | 0.942 | 0.352 | 0.937 |
| sentiment | 0.60 | 0.50 | medium | 1825 | 0.275 | 0.832 | 0.270 | 0.817 |
| sentiment | 0.60 | 0.50 | low | 2525 | 0.380 | 0.579 | 0.378 | 0.593 |
| aspect | 0.50 | 0.35 | high | 4720 | 0.710 | 0.981 | 0.713 | 0.981 |
| aspect | 0.50 | 0.35 | medium | 1196 | 0.180 | 0.816 | 0.181 | 0.812 |
| aspect | 0.50 | 0.35 | low | 729 | 0.110 | 0.554 | 0.106 | 0.548 |

Band counts pool the three repeats, so every review appears three times and a band of 30 rows holds about 10 distinct reviews. Thresholds (high / medium) computed on each repeat alone: sentiment: 0.60 / 0.50, 0.60 / 0.50, 0.60 / 0.50; aspect: 0.45 / 0.35, 0.50 / 0.40, 0.50 / 0.35.

Caveats: the final models are trained on the whole development partition (about 25% more rows than a CV training fold), so their probabilities may be somewhat higher and the share of high-certainty answers larger. The real accuracy per level is measured on the test partition in M5 and reported as found. The fresh-seed columns use folds that played no role in setting the thresholds.

## 3. Expected test range

Computed after the configurations were frozen, from CV repeats with fresh seeds 100, 101, 102 (grouped and stratified like the selection folds, never used for any choice): mean ± 2 standard deviations of the 15 fold scores.

| task | configuration | mean Macro-F1 | sd | expected range |
|---|---|---|---|---|
| sentiment | final `m4_logreg_C0.3` | 0.6564 | 0.0248 | [0.607, 0.706] |
| sentiment | reference `logreg_word11` | 0.5316 | 0.0271 | [0.477, 0.586] |
| aspect | final `m4_logreg_C1` | 0.9075 | 0.0140 | [0.879, 0.936] |
| aspect | reference `logreg_word11` | 0.7948 | 0.0181 | [0.759, 0.831] |

Joint accuracy (both labels right, final configurations): mean 0.7054, sd 0.0202, expected range [0.665, 0.746].

A CV fold holds about 443 reviews and is predicted by a model trained on about 1,772; the test partition holds 554 and is predicted by a model trained on all 2,215, so the test score may sit slightly above the CV mean. The sampling part of the fold sd shrinks by about sqrt(443/554) = 0.89 on the larger test partition, so the range is slightly conservative. It is recorded for comparison only: it is not a pass mark, and no result outside it leads to another test run.

## 4. Clean-tree rerun of the cited rows

The configurations cited here were run again on the selection folds from a committed tree and compared fold by fold with the logged rows they were selected from.

| task | configuration | original commit | rerun commit | original mean | rerun mean | largest fold difference |
|---|---|---|---|---|---|---|
| sentiment | `dummy_most_frequent` | b101ad8+dirty | 622b560 | 0.2601 | 0.2601 | 0.0e+00 |
| sentiment | `dummy_stratified` | b101ad8+dirty | 622b560 | 0.3479 | 0.3479 | 0.0e+00 |
| sentiment | `logreg_word11` | b101ad8+dirty | 622b560 | 0.5236 | 0.5236 | 0.0e+00 |
| sentiment | `a6_balanced` | 83132aa | 622b560 | 0.6560 | 0.6560 | 0.0e+00 |
| sentiment | `m4_logreg_C0.3` | db1d5db | 622b560 | 0.6529 | 0.6529 | 0.0e+00 |
| aspect | `dummy_most_frequent` | b101ad8+dirty | 622b560 | 0.0963 | 0.0963 | 0.0e+00 |
| aspect | `dummy_stratified` | b101ad8+dirty | 622b560 | 0.2016 | 0.2016 | 0.0e+00 |
| aspect | `logreg_word11` | b101ad8+dirty | 622b560 | 0.7923 | 0.7923 | 0.0e+00 |
| aspect | `a6_balanced` | 83132aa | 622b560 | 0.9069 | 0.9069 | 0.0e+00 |
| aspect | `m4_logreg_C1` | db1d5db | c546512 | 0.9069 | 0.9069 | 0.0e+00 |

## 5. One-shot test procedure (M5)

1. Before it opens the test partition, `python -m arabic_reviews final` rebuilds the protocol from code and the development partition (configurations, pipeline parameters, certainty thresholds from the out-of-fold predictions, data hashes, versions; no rerun rows) and stops unless the fingerprint equals the one above.
2. For each task it fits the frozen configuration, the reference model and both dummy baselines on the whole development partition.
3. It reads the held-out test partition once, predicts it once per model, and records the access (time, commit, fingerprint, file hash) in `reports/test_access_log.json`. A second run is refused unless an explicit flag and a written methodological reason are given; the first result stays the reported one.
4. Reported per task: Macro-F1 (primary), per-class precision / recall / F1, balanced accuracy, accuracy and the confusion matrix; joint accuracy over both tasks; accuracy and share per certainty level; the position of the test Macro-F1 relative to the expected range above.
5. Uncertainty: 95% percentile bootstrap intervals with 2000 resamples (seed 0). Resampling unit: group_id: draw as many near-duplicate groups as the test partition has, with replacement (rows of a group are not independent). Metrics: Macro-F1 and per-class F1 with the fixed label list and zero_division=0, as in the CV. Paired comparisons: final minus reference and final minus each dummy, on the same resamples: interval of the difference and share of resamples with a difference <= 0. Decision: final beats a dummy when the lower bound of the paired difference is > 0; versus the reference the result is 'above' (lower bound > 0), 'not distinguishable' (interval contains 0) or 'below' (upper bound < 0); success requires beating both dummies and not 'below' the reference. No claim is made from very small cells (for example prices x neutral).
6. Errors on the test partition are described, but no model, threshold or feature is changed after the test has been read.
