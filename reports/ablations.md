# Feature-engineering ablations (development CV only)

## Axis 1 - representation

Rule: mean paired Macro-F1 gain >= 0.01 and gain in >= 11 of 15 folds.

### sentiment

| candidate | vs current winner | mean gain | folds won | decision |
|---|---|---|---|---|
| logreg_word12 | logreg_word11 | -0.0382 | 0/15 | rejected |
| logreg_char25 | logreg_word11 | +0.0262 | 11/15 | accepted |
| logreg_word11+char25 | logreg_char25 | +0.0506 | 13/15 | accepted |

Winner for sentiment: `logreg_word11+char25`

### aspect

| candidate | vs current winner | mean gain | folds won | decision |
|---|---|---|---|---|
| logreg_word12 | logreg_word11 | -0.0327 | 0/15 | rejected |
| logreg_char25 | logreg_word11 | +0.0992 | 15/15 | accepted |
| logreg_word11+char25 | logreg_char25 | -0.0010 | 7/15 | rejected |

Winner for aspect: `logreg_char25`

