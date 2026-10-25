# Exploratory data analysis — development partition only

Written by `python -m arabic_reviews eda` from `split.load_dev()`; the held-out test partition is not read. Counts are rows of the dev partition.

Missing values, exact and normalised duplicates and label conflicts are reported for the whole validated file in [`data_quality.md`](data_quality.md); near-duplicate grouping and the partition balance in [`split.md`](split.md).

Rows: **2215**, near-duplicate groups: 2090.

## 1. Labels

| aspect \ sentiment | ايجابي | سلبي | محايد | total |
|---|---|---|---|---|
| المأكولات | 251 | 47 | 38 | 336 |
| المشروبات | 548 | 105 | 49 | 702 |
| الخدمة | 216 | 240 | 56 | 512 |
| الجلسات | 291 | 83 | 36 | 410 |
| الاسعار | 111 | 120 | 24 | 255 |
| total | 1417 | 595 | 203 | 2215 |

Neutral share per aspect: المأكولات 11.3%, المشروبات 7.0%, الخدمة 10.9%, الجلسات 8.8%, الاسعار 9.4%.

## 2. Text length

| sentiment | rows | chars median | chars p90 | words median | words p90 | words max |
|---|---|---|---|---|---|---|
| ايجابي | 1417 | 27 | 68 | 5 | 12 | 54 |
| سلبي | 595 | 43 | 124.6 | 8 | 24 | 65 |
| محايد | 203 | 41 | 106 | 8 | 19.8 | 50 |

| aspect | rows | chars median | chars p90 | words median | words p90 | words max |
|---|---|---|---|---|---|---|
| المأكولات | 336 | 32.5 | 91 | 6 | 17 | 59 |
| المشروبات | 702 | 29 | 77.9 | 5 | 15 | 60 |
| الخدمة | 512 | 39 | 121.9 | 7 | 23 | 65 |
| الجلسات | 410 | 29.5 | 81 | 5 | 15 | 48 |
| الاسعار | 255 | 31 | 73.6 | 6 | 14 | 41 |

Reviews with at most 3 words: 20.6% of dev.

## 3. Negation and emoji by sentiment

| sentiment | % with a negation particle | % with a negation particle other than ما | % with an emoji |
|---|---|---|---|
| ايجابي | 8.6 | 6.2 | 6.9 |
| سلبي | 33.3 | 26.4 | 3.5 |
| محايد | 28.1 | 21.7 | 4.4 |

Negation particles: `text.NEGATIONS` (14 forms), matched as whole tokens after full normalisation. `ما` is counted as a negation in the first column although it also has non-negating uses (ما شاء الله, exclamative ما احلى), so that column overstates negation; the second column leaves `ما` out.

## 4. Most distinctive word 1–2-grams per class (chi², one-vs-rest)

Fitted on the dev partition for description only; model features are fitted inside each CV fold.

| sentiment | top n-grams |
|---|---|
| ايجابي | ممتاز، رائع، جميل، قهوه، مكان، احلي، افضل، جميله |
| سلبي | غير، مبالغ، للاسف، سيئ، سيئه، سيء، غالي، الفرع |
| محايد | بس، جيد، عادي، لاكن، ممتاز بس، ولكن، جيد ولكن، مقبول |

| aspect | top n-grams |
|---|---|
| المأكولات | الدونات، دونات، كروسان، والدونات، والكوكيز، لذيذه، الكوكيز، لذيذ |
| المشروبات | القهوه، قهوه، قهوتهم، المشروبات، موكا، لاتيه، وايت، البارده |
| الخدمة | خدمه، الخدمه، الموظفين، الطلب، زحمه، السيارات، ساعه، سريعه |
| الجلسات | مكان، المكان، جلسات، الجلسات، هادئ، مريحه، جميل، هادي |
| الاسعار | الاسعار، اسعار، اسعاره، مبالغ، غالي، مبالغ فيها، اسعارهم، مرتفعه |

## 5. How often does a review mention more than one aspect?

Estimated with a small hand-written list of word stems per aspect (`eda.ASPECT_CUES`), matched on normalised text. This is a rough indicator for the single-label limitation, not a measurement.

| aspects with a cue | 0 | 1 | 2 | 3+ |
|---|---|---|---|---|
| reviews | 378 | 1605 | 219 | 13 |

- **10.5%** of dev reviews contain cues for two or more aspects; for these, the labelled aspect is among the cued ones in 96.1% of cases.
- Where exactly one aspect is cued, it agrees with the label in 94.8% of cases (a sanity check of the cue list).

## 6. Examples (first two dev rows per aspect, by rid)

| rid | aspect | sentiment | text |
|---|---|---|---|
| 1 | المأكولات | سلبي | منيو الطعام سيء |
| 3 | المأكولات | ايجابي | افضل دونات تبدا فيها يومك |
| 4 | المشروبات | ايجابي | خيار ممتاز للقهوه |
| 5 | المشروبات | محايد | خساره طلبت بلاك كوفي مقبول |
| 21 | الخدمة | سلبي | وخدمه غير مرضية |
| 32 | الخدمة | ايجابي | أحب خدمة السيارات لديهم |
| 17 | الجلسات | ايجابي | احببته، مكان يستحق الزيارة |
| 18 | الجلسات | ايجابي | أفضل مكان للاستمتاع مع الأصدقاء . |
| 22 | الاسعار | محايد | اسعار مرتفعه وارتفاع غير مقبول |
| 25 | الاسعار | سلبي | سعر غالي مبالغ فيه |

## 7. Observations that inform M3

- **Neutral is rare (9.2%) and lexically hedged**: its most distinctive terms are concessive or lukewarm words (بس، جيد، عادي، لاكن، ممتاز بس، ولكن، جيد ولكن، مقبول). This supports Macro-F1 as the primary metric, the class-weight axis and word bigrams. Some neutral rows read as negative (rid 22), matching the label noise noted in `docs/PROBLEM_FRAMING.md`; neutral recall will be the weakest point to watch.
- **Negation carries sentiment signal**: a negation particle appears in 33.3% of negative and 28.1% of neutral reviews against 8.6% of positive ones (26.4% / 21.7% / 6.2% without `ما`); غير is among the top negative terms. Negations must survive preprocessing; stop-word removal is measured, not assumed.
- **Aspect cues are lexical but split by clitics and spelling**: الدونات/دونات/والدونات, قهوه/القهوه/قهوتهم, اسعار/الاسعار/اسعاره. Character n-grams (`char_wb`) or a word+character union are the expected way to share evidence across these forms; this is the first ablation axis.
- **Reviews are short**: median 5–8 words by sentiment, and 20.6% have at most 3 words. Negative and neutral reviews are longer than positive ones (median 8 and 8 vs 5 words), so length is a candidate simple feature for sentiment (axis 5), to be kept only if it helps under the acceptance rule.
- **Emoji lean positive** (6.9% of positive vs 3.5% negative and 4.4% neutral reviews): emoji are kept as tokens in the symbols axis rather than stripped by default.
- **Aspect and sentiment are correlated**: ايجابي is the majority for المأكولات 74.7%, المشروبات 78.1%, الجلسات 71.0%; سلبي is the largest class for الخدمة 46.9%, الاسعار 47.1% (not a majority; runner-up الخدمة: ايجابي 42.2%, الاسعار: ايجابي 43.5%). The two tasks stay separate models; the joint-correct rate is reported as a secondary metric.
- **Single-label limitation is real but limited**: about 10.5% of reviews mention two or more aspects by the cue estimate, and the labelled aspect is almost always one of them (96.1%). Multi-aspect reviews are therefore expected to explain only part of the aspect errors; error analysis in M4 will count them.

## Figures

- `reports/figures/dev_label_matrix.png`
- `reports/figures/dev_length_by_sentiment.png`
