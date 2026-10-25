# Model families and limited tuning (development CV only)

Same 3 x 5 grouped CV, same rule: mean paired Macro-F1 gain >= 0.01 and gain in >= 11 of 15 folds. Features are the frozen M3 sets.

## sentiment

Reference (M3 final): `a6_balanced`

### Stage A - model families

| candidate | vs | mean gain | folds won | decision |
|---|---|---|---|---|
| m4_linsvm | a6_balanced | -0.0217 | 2/15 | rejected |
| m4_cnb | a6_balanced | -0.1707 | 0/15 | rejected |

Family carried to tuning: `logreg`

### Stage B - C grid (one-standard-error rule, simplest first)

| C | mean Macro-F1 | std |
|---|---|---|
| 0.1 | 0.6488 | 0.0244 |
| 0.3 (one-SE pick) **(used)** | 0.6529 | 0.0244 |
| 1 | 0.6560 | 0.0235 |
| 3 | 0.6525 | 0.0220 |
| 10 | 0.6485 | 0.0244 |

The one-standard-error pick is C = 0.3. It is not weaker-regularised than the default, so no further check applies. The standard error is the sd of the 15 fold scores over sqrt(15); the three repeats reuse the same reviews, so this understates the error, and with sqrt(5) the pick would move one step towards stronger regularisation. The rule was committed before the grid was run and is not changed afterwards.

| tf-idf option | vs | mean gain | folds won | decision |
|---|---|---|---|---|
| m4_logreg_C0.3+min_df2 | m4_logreg_C0.3 | -0.0057 | 5/15 | rejected |
| m4_logreg_C0.3+sublinear | m4_logreg_C0.3 | +0.0000 | 9/15 | rejected |

Tuned configuration `m4_logreg_C0.3` vs M3 reference: -0.0031 mean, 6/15 folds (does not pass the acceptance rule).

## aspect

Reference (M3 final): `a6_balanced`

### Stage A - model families

| candidate | vs | mean gain | folds won | decision |
|---|---|---|---|---|
| m4_linsvm | a6_balanced | +0.0070 | 11/15 | rejected |
| m4_cnb | a6_balanced | -0.0063 | 5/15 | rejected |

Family carried to tuning: `logreg`

### Stage B - C grid (one-standard-error rule, simplest first)

| C | mean Macro-F1 | std |
|---|---|---|
| 0.1 | 0.8854 | 0.0092 |
| 0.3 | 0.8942 | 0.0113 |
| 1 **(used)** | 0.9069 | 0.0157 |
| 3 (one-SE pick) | 0.9101 | 0.0162 |
| 10 | 0.9117 | 0.0132 |

The one-standard-error pick is C = 3. It is weaker-regularised than the default C = 1, i.e. it adds complexity, so it must also pass the acceptance rule: +0.0032 mean, 10/15 folds; it fails, so the default C = 1 is used. The standard error is the sd of the 15 fold scores over sqrt(15); the three repeats reuse the same reviews, so this understates the error, and with sqrt(5) the pick would move one step towards stronger regularisation. The rule was committed before the grid was run and is not changed afterwards.

| tf-idf option | vs | mean gain | folds won | decision |
|---|---|---|---|---|
| m4_logreg_C1+min_df2 | m4_logreg_C1 | +0.0020 | 12/15 | rejected |
| m4_logreg_C1+sublinear | m4_logreg_C1 | +0.0028 | 10/15 | rejected |

Tuned configuration `m4_logreg_C1` vs M3 reference: +0.0000 mean, 0/15 folds (does not pass the acceptance rule).

## Limits of this search

- Families were compared at their default regularisation only; a tuned linear SVM might do slightly better than in Stage A, where it did not pass the rule for either task.
- `min_df=2` shrinks the vocabulary and is arguably the simpler option; it was nevertheless treated as a candidate change that has to pass the rule, as declared before the run. Its gains were within noise for both tasks.
