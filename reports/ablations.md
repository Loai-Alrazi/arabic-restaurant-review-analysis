# Feature-engineering ablations (development CV only)

## Axis 1 - representation

Rule: mean paired Macro-F1 gain >= 0.01 and gain in >= 11 of 15 folds.

### sentiment

| candidate | vs current winner | mean gain | folds won | decision |
|---|---|---|---|---|
| logreg_word12 | logreg_word11 | -0.0381 | 0/15 | rejected |
| logreg_char25 | logreg_word11 | +0.0263 | 11/15 | accepted |
| logreg_word11+char25 | logreg_char25 | +0.0507 | 13/15 | accepted |

Winner for sentiment: `logreg_word11+char25`

### aspect

| candidate | vs current winner | mean gain | folds won | decision |
|---|---|---|---|---|
| logreg_word12 | logreg_word11 | -0.0327 | 0/15 | rejected |
| logreg_char25 | logreg_word11 | +0.0992 | 15/15 | accepted |
| logreg_word11+char25 | logreg_char25 | -0.0010 | 7/15 | rejected |

Winner for aspect: `logreg_char25`

## Axes 2-6

Each candidate is the current winner plus one change, compared on the same 15 folds.

### sentiment

| axis | candidate | vs current winner | mean gain | folds won | decision |
|---|---|---|---|---|---|
| Axis 2 - normalisation | a2_light | logreg_word11+char25 | +0.0020 | 7/15 | rejected |
| Axis 2 - normalisation | a2_full | logreg_word11+char25 | +0.0046 | 10/15 | rejected |
| Axis 3 - symbols and digits | a3_symbols_separate | logreg_word11+char25 | +0.0003 | 5/15 | rejected |
| Axis 3 - symbols and digits | a3_symbols_remove | logreg_word11+char25 | +0.0005 | 7/15 | rejected |
| Axis 3 - symbols and digits | a3_digits_token | logreg_word11+char25 | +0.0019 | 9/15 | rejected |
| Axis 4 - negation and stopwords | a4_negation | logreg_word11+char25 | +0.0044 | 10/15 | rejected |
| Axis 4 - negation and stopwords | a4_stopwords | logreg_word11+char25 | -0.0067 | 6/15 | rejected |
| Axis 5 - text statistics | a5_textstats | logreg_word11+char25 | -0.0144 | 2/15 | rejected |
| Axis 6 - class weighting | a6_balanced | logreg_word11+char25 | +0.0555 | 15/15 | accepted |

Final feature set for sentiment: `{"class_weight": "balanced", "normalisation": "none", "representation": "word11+char25"}`

### aspect

| axis | candidate | vs current winner | mean gain | folds won | decision |
|---|---|---|---|---|---|
| Axis 2 - normalisation | a2_light | logreg_char25 | +0.0005 | 8/15 | rejected |
| Axis 2 - normalisation | a2_full | logreg_char25 | +0.0029 | 11/15 | rejected |
| Axis 3 - symbols and digits | a3_symbols_separate | logreg_char25 | -0.0003 | 4/15 | rejected |
| Axis 3 - symbols and digits | a3_symbols_remove | logreg_char25 | +0.0003 | 6/15 | rejected |
| Axis 3 - symbols and digits | a3_digits_token | logreg_char25 | -0.0010 | 7/15 | rejected |
| Axis 4 - negation and stopwords | a4_negation | logreg_char25 | +0.0005 | 10/15 | rejected |
| Axis 4 - negation and stopwords | a4_stopwords | logreg_char25 | +0.0005 | 7/15 | rejected |
| Axis 5 - text statistics | a5_textstats | logreg_char25 | -0.0058 | 2/15 | rejected |
| Axis 6 - class weighting | a6_balanced | logreg_char25 | +0.0154 | 13/15 | accepted |

Final feature set for aspect: `{"class_weight": "balanced", "normalisation": "none", "representation": "char25"}`

