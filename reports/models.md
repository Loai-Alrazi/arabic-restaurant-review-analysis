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
| 0.3 **(picked)** | 0.6529 | 0.0244 |
| 1 | 0.6560 | 0.0235 |
| 3 | 0.6525 | 0.0220 |
| 10 | 0.6485 | 0.0244 |

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
| 1 | 0.9069 | 0.0157 |
| 3 **(picked)** | 0.9101 | 0.0162 |
| 10 | 0.9117 | 0.0132 |

| tf-idf option | vs | mean gain | folds won | decision |
|---|---|---|---|---|
| m4_logreg_C3+min_df2 | m4_logreg_C3 | +0.0001 | 7/15 | rejected |
| m4_logreg_C3+sublinear | m4_logreg_C3 | +0.0024 | 8/15 | rejected |

Tuned configuration `m4_logreg_C3` vs M3 reference: +0.0032 mean, 10/15 folds (does not pass the acceptance rule).
