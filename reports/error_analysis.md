# Out-of-fold error analysis (development partition)

## sentiment

Model: `m4_logreg_C0.3` (logreg), repeat 0 out-of-fold predictions on 2215 development reviews; Macro-F1 0.6506, accuracy 0.7711, errors 507.

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| ايجابي | 0.899 | 0.827 | 0.861 | 1417 |
| سلبي | 0.675 | 0.761 | 0.716 | 595 |
| محايد | 0.346 | 0.409 | 0.375 | 203 |

Confusion matrix (rows = true, columns = predicted):

| true \ pred | ايجابي | سلبي | محايد |
|---|---|---|---|
| ايجابي | 1172 | 153 | 92 |
| سلبي | 77 | 453 | 65 |
| محايد | 55 | 65 | 83 |

## aspect

Model: `m4_logreg_C1` (logreg), repeat 0 out-of-fold predictions on 2215 development reviews; Macro-F1 0.9048, accuracy 0.9020, errors 217.

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| المأكولات | 0.913 | 0.902 | 0.907 | 336 |
| المشروبات | 0.937 | 0.907 | 0.922 | 702 |
| الخدمة | 0.826 | 0.906 | 0.864 | 512 |
| الجلسات | 0.900 | 0.883 | 0.892 | 410 |
| الاسعار | 0.971 | 0.910 | 0.939 | 255 |

Confusion matrix (rows = true, columns = predicted):

| true \ pred | المأكولات | المشروبات | الخدمة | الجلسات | الاسعار |
|---|---|---|---|---|---|
| المأكولات | 303 | 10 | 18 | 2 | 3 |
| المشروبات | 15 | 637 | 30 | 16 | 4 |
| الخدمة | 11 | 16 | 464 | 21 | 0 |
| الجلسات | 2 | 9 | 37 | 362 | 0 |
| الاسعار | 1 | 8 | 13 | 1 | 232 |

## Manual classification of real errors

A seeded random sample of out-of-fold errors (repeat 0, `oof_error_sample`, seed 7; 45 sentiment errors of `m4_logreg_C0.3`, 30 aspect errors of `m4_logreg_C3`) was read, and each error was assigned one primary category from its text, gold label and prediction. The per-review categories (without texts) are in `reports/error_categories.csv`. The categories are judgements, not measurements, and the samples are small.

| category | sentiment (n=45) | aspect (n=30) |
|---|---|---|
| lexical or semantic variety not captured by n-grams | 10 (22%) | 9 (30%) |
| several aspects / mixed polarity in one text | 9 (20%) | 12 (40%) |
| borderline neutral | 6 (13%) | 0 |
| dialect or irregular spelling | 5 (11%) | 0 |
| label noise (gold label looks wrong or debatable) | 5 (11%) | 6 (20%) |
| negation | 4 (9%) | 0 |
| sarcasm or implicit opinion | 4 (9%) | 0 |
| too short or uninformative | 1 (2%) | 3 (10%) |
| other | 1 (2%) | 0 |

Pre-declared gate for the optional pre-trained sentence-vector experiment: the category "lexical or semantic variety not captured by n-grams" must be the largest and at least 30% of the sample.
- sentiment: 22%, the largest category -> gate stays closed.
- aspect: 30%, not the largest category -> gate stays closed.

The aspect sample was drawn from the errors of `m4_logreg_C3`, the configuration chosen before the complexity check on the regularisation grid was added; the final configuration is `m4_logreg_C1` (`reports/models.md`). The gate concerns the representation, which both share, not the regularisation strength.
