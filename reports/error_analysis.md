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

Model: `m4_logreg_C3` (logreg), repeat 0 out-of-fold predictions on 2215 development reviews; Macro-F1 0.9098, accuracy 0.9052, errors 210.

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| المأكولات | 0.942 | 0.911 | 0.926 | 336 |
| المشروبات | 0.924 | 0.915 | 0.919 | 702 |
| الخدمة | 0.833 | 0.896 | 0.864 | 512 |
| الجلسات | 0.901 | 0.885 | 0.893 | 410 |
| الاسعار | 0.975 | 0.922 | 0.948 | 255 |

Confusion matrix (rows = true, columns = predicted):

| true \ pred | المأكولات | المشروبات | الخدمة | الجلسات | الاسعار |
|---|---|---|---|---|---|
| المأكولات | 306 | 11 | 15 | 2 | 2 |
| المشروبات | 9 | 642 | 30 | 17 | 4 |
| الخدمة | 9 | 25 | 459 | 19 | 0 |
| الجلسات | 0 | 9 | 38 | 363 | 0 |
| الاسعار | 1 | 8 | 9 | 2 | 235 |

## Manual classification of real errors

A seeded random sample of out-of-fold errors (seed 7: 45 sentiment, 30 aspect; 75 in total) was read and given one primary category each by a reader who saw only the text, the gold label and the prediction. The categories are judgements, not measurements, and the samples are small.

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

Reading: the weak class is neutral (F1 0.375), whose errors are mostly borderline polarity and mixed texts. Aspect errors are dominated by reviews that touch more than one aspect, plus label noise. The category "lexical variety" is not the largest in either task and is below 30% for sentiment, so the pre-declared condition for the optional pre-trained sentence-vector experiment is not met for either task; it is not run (for aspect, the 30% share is not the largest category).
