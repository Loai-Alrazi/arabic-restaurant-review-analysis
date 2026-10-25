# Final evaluation protocol (frozen before the held-out test is read)

This file and `reports/protocol.json` fix every choice that the one-shot test in M5 depends on. All numbers come from the development partition only; the held-out test partition has not been read by any step up to here (experiment code can only call `split.load_dev()`).

- Protocol fingerprint (configurations + certainty thresholds): `250d1d7cbaa75a17d8d162c8028b44bdd438323ae4af9e17716c4ebd49c12131`
- Development partition: 2215 reviews, `data/processed/dev.csv` SHA-256 `f0d6dc75ebd542d24cb04993bd4a492c13bd14433d91b05b58828acbd795c708`
- Split file `data/splits/split_v1.csv` SHA-256 `6a8a686fc143992313443060c5106ed80dd78a09013e85665bbeea39930a70f0`

## 1. Frozen configurations

| task | role | configuration | model | features | normalisation | class weight | parameters | selection CV Macro-F1 (3 x 5) |
|---|---|---|---|---|---|---|---|---|
| sentiment | final | `m4_logreg_C0.3` | logreg | word11+char25 TF-IDF | none | balanced | `{"C": 0.3}` | 0.6529 ± 0.0244 |
| sentiment | reference | `logreg_word11` | logreg | word11 TF-IDF | none | None | `{}` | 0.5236 ± 0.0285 |
| aspect | final | `m4_logreg_C3` | logreg | char25 TF-IDF | none | balanced | `{"C": 3}` | 0.9101 ± 0.0162 |
| aspect | reference | `logreg_word11` | logreg | word11 TF-IDF | none | None | `{}` | 0.7923 ± 0.0157 |

TF-IDF uses scikit-learn defaults (no `min_df`, no `sublinear_tf`); `word11` is word unigrams, `char25` is character 2-5 grams inside word boundaries, `+` joins both in one feature union. Logistic regression uses `max_iter=2000` and `random_state=0`. The dummy baselines (`most_frequent`, `stratified`) are reported next to the models in M5.

How the configurations were chosen (all on the same 3 x 5 grouped CV; rule: mean paired Macro-F1 gain >= 0.01 and a gain in >= 11 of 15 folds):

- Representation and preprocessing: `reports/ablations.md`. Sentiment uses word unigrams + character 2-5 grams, aspect character 2-5 grams. No normalisation, symbol, digit, negation, stopword or text-statistic option passed the rule; balanced class weights passed for both tasks.
- Model family and regularisation: `reports/models.md`. Linear SVM and Complement NB did not pass the rule, so logistic regression stays. The regularisation strength comes from the pre-declared C grid with the one-standard-error rule (sentiment C = 0.3, aspect C = 3). Both picks differ from the default C = 1 by less than the fold noise (sentiment -0.003, aspect +0.003 mean Macro-F1). They are kept because the selection procedure was fixed before the grid was run, not because they are better.
- The optional pre-trained sentence-vector experiment was not run: its pre-declared condition was not met (`reports/error_analysis.md`).

## 2. Certainty levels

Score: the highest class probability of the task's logistic-regression pipeline. Thresholds come from out-of-fold predictions of the frozen configuration on the three selection repeats (seeds 42-44). Scores are cut into bands of width 0.05; a level starts at the smallest band edge from which every band with at least 30 out-of-fold rows is right at least as often as its target (high: 90%, medium: 70%); lower scores are low. The interface shows the level in words, never the raw probability.

| task | high from | medium from | level | selection OOF rows | share | accuracy | fresh-seed OOF share | accuracy |
|---|---|---|---|---|---|---|---|---|
| sentiment | 0.60 | 0.50 | high | 2295 | 0.345 | 0.942 | 0.352 | 0.937 |
| sentiment | 0.60 | 0.50 | medium | 1825 | 0.275 | 0.832 | 0.270 | 0.817 |
| sentiment | 0.60 | 0.50 | low | 2525 | 0.380 | 0.579 | 0.378 | 0.593 |
| aspect | 0.60 | 0.40 | high | 4928 | 0.742 | 0.980 | 0.744 | 0.983 |
| aspect | 0.60 | 0.40 | medium | 1139 | 0.171 | 0.794 | 0.170 | 0.773 |
| aspect | 0.60 | 0.40 | low | 578 | 0.087 | 0.498 | 0.085 | 0.514 |

Caveats: the final models are trained on the whole development partition (about 25% more rows than a CV training fold), so their probabilities may be somewhat higher and the share of high-certainty answers larger. The real accuracy per level is measured on the test partition in M5 and reported as found. The fresh-seed columns use folds that played no role in setting the thresholds.

## 3. Expected test range

Computed after the configurations were frozen, from CV repeats with fresh seeds 100, 101, 102 (grouped and stratified like the selection folds, never used for any choice): mean ± 2 standard deviations of the 15 fold scores.

| task | configuration | mean Macro-F1 | sd | expected range |
|---|---|---|---|---|
| sentiment | final `m4_logreg_C0.3` | 0.6564 | 0.0248 | [0.607, 0.706] |
| sentiment | reference `logreg_word11` | 0.5316 | 0.0271 | [0.477, 0.586] |
| aspect | final `m4_logreg_C3` | 0.9106 | 0.0144 | [0.882, 0.939] |
| aspect | reference `logreg_word11` | 0.7948 | 0.0181 | [0.759, 0.831] |

Joint accuracy (both labels right, final configurations): mean 0.7085, sd 0.0186, expected range [0.671, 0.746].

A CV fold holds about 443 reviews and is predicted by a model trained on about 1,770; the test partition holds 554 and is predicted by a model trained on all 2,215, so the test score may sit slightly above the CV mean. The range is recorded for comparison only: it is not a pass mark, and no result outside it leads to another test run.

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
| aspect | `m4_logreg_C3` | db1d5db | 622b560 | 0.9101 | 0.9101 | 0.0e+00 |

## 5. One-shot test procedure (M5)

1. `python -m arabic_reviews final` recomputes the fingerprint from the code and this protocol and stops if they differ.
2. For each task it fits the frozen configuration, the reference model and both dummy baselines on the whole development partition.
3. It reads the held-out test partition once, predicts it once per model, and records the access (time, commit, fingerprint, file hash) in `reports/test_access_log.json`. A second run is refused unless an explicit flag and a written methodological reason are given; the first result stays the reported one.
4. Reported per task: Macro-F1 (primary), per-class precision / recall / F1, balanced accuracy, accuracy and the confusion matrix; joint accuracy over both tasks; accuracy and share per certainty level; the position of the test Macro-F1 relative to the expected range above.
5. Uncertainty: 95% percentile bootstrap intervals for Macro-F1 and every per-class F1 with 2000 resamples (seed 0), resampling whole near-duplicate groups (`group_id`) because rows of a group are not independent, and a paired bootstrap of the Macro-F1 difference between the final configuration and the reference model on the same resamples. No claim is made from very small cells (for example prices x neutral).
6. Errors on the test partition are described, but no model, threshold or feature is changed after the test has been read.
