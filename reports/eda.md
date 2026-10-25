# Exploratory data analysis — development partition only

Written by `python -m arabic_reviews eda` from `split.load_dev()`; the held-out test partition is not read. Counts are rows of the dev partition.

Rows: **2215**, near-duplicate groups: 2126.

## 1. Labels

| aspect \ sentiment | ايجابي | سلبي | محايد | total |
|---|---|---|---|---|
| المأكولات | 251 | 47 | 38 | 336 |
| المشروبات | 548 | 106 | 49 | 703 |
| الخدمة | 216 | 239 | 56 | 511 |
| الجلسات | 291 | 83 | 36 | 410 |
| الاسعار | 111 | 120 | 24 | 255 |
| total | 1417 | 595 | 203 | 2215 |

Neutral share per aspect: المأكولات 11.3%, المشروبات 7.0%, الخدمة 11.0%, الجلسات 8.8%, الاسعار 9.4%.

## 2. Text length

| sentiment | rows | chars median | chars p90 | words median | words p90 | words max |
|---|---|---|---|---|---|---|
| ايجابي | 1417 | 27 | 72 | 5 | 13 | 54 |
| سلبي | 595 | 42 | 116.6 | 8 | 23 | 61 |
| محايد | 203 | 38 | 101 | 7 | 18 | 38 |

| aspect | rows | chars median | chars p90 | words median | words p90 | words max |
|---|---|---|---|---|---|---|
| المأكولات | 336 | 33.5 | 87 | 6 | 16 | 59 |
| المشروبات | 703 | 29 | 76 | 5 | 15 | 50 |
| الخدمة | 511 | 41 | 123 | 8 | 23 | 61 |
| الجلسات | 410 | 29 | 75 | 5 | 13.1 | 32 |
| الاسعار | 255 | 31 | 74 | 5 | 14 | 41 |

Reviews with at most 3 words: 19.8% of dev.

## 3. Negation and emoji by sentiment

| sentiment | % with a negation particle | % with an emoji |
|---|---|---|
| ايجابي | 8.9 | 7.4 |
| سلبي | 32.4 | 4.0 |
| محايد | 27.1 | 2.5 |

Negation particles: `text.NEGATIONS` (14 forms), matched as whole tokens after full normalisation.

## 4. Most distinctive word 1–2-grams per class (chi², one-vs-rest)

Fitted on the dev partition for description only; model features are fitted inside each CV fold.

| sentiment | top n-grams |
|---|---|
| ايجابي | ممتاز، جميل، رائع، مكان، قهوه، افضل، رائعه، احلي |
| سلبي | للاسف، الفرع، سيء، غير، سيئه، سيئ، غالي، الطلب |
| محايد | جيد، بس، لاكن، عادي، مقبول، ولكن، جيد لكن، يكون |

| aspect | top n-grams |
|---|---|
| المأكولات | الدونات، دونات، كروسان، والدونات، والكوكيز، لذيذه، الكوكيز، لذيذ |
| المشروبات | القهوه، قهوه، قهوتهم، موكا، المشروبات، لاتيه، وايت، البارده |
| الخدمة | الخدمه، خدمه، الموظفين، ساعه، الطلب، سريعه، السيارات، زحمه |
| الجلسات | مكان، المكان، جلسات، الجلسات، جميل، جلساته، مكان جميل، هادي |
| الاسعار | الاسعار، اسعار، اسعاره، غالي، مبالغ، اسعارهم، مرتفعه، غاليه |

## 5. How often does a review mention more than one aspect?

Estimated with a small hand-written list of word stems per aspect (`eda.ASPECT_CUES`), matched on normalised text. This is a rough indicator for the single-label limitation, not a measurement.

| aspects with a cue | 0 | 1 | 2 | 3+ |
|---|---|---|---|---|
| reviews | 391 | 1607 | 205 | 12 |

- **9.8%** of dev reviews contain cues for two or more aspects; for these, the labelled aspect is among the cued ones in 97.2% of cases.
- Where exactly one aspect is cued, it agrees with the label in 95.3% of cases (a sanity check of the cue list).

## 6. Examples (first two dev rows per aspect, by rid)

| rid | aspect | sentiment | text |
|---|---|---|---|
| 0 | المأكولات | ايجابي | الاكل كان مناسب والسانويش كان لذيذ |
| 1 | المأكولات | سلبي | منيو الطعام سيء |
| 5 | المشروبات | محايد | خساره طلبت بلاك كوفي مقبول |
| 6 | المشروبات | ايجابي | حلو مطعم وسيع يبيع أنواع القهوه الذيذه |
| 21 | الخدمة | سلبي | وخدمه غير مرضية |
| 60 | الخدمة | سلبي | - بطء استلام الطلب لكن تسليمه سريع جدًا < الطلبات الداخلية ويبدو ان تركيزهم على السيارات اكبر |
| 17 | الجلسات | ايجابي | احببته، مكان يستحق الزيارة |
| 18 | الجلسات | ايجابي | أفضل مكان للاستمتاع مع الأصدقاء . |
| 22 | الاسعار | محايد | اسعار مرتفعه وارتفاع غير مقبول |
| 23 | الاسعار | سلبي | الاسعار غاليه |

## 7. Observations that inform M3

- **Neutral is rare (9.2%) and lexically hedged**: its distinctive terms are concessive or lukewarm words (جيد، بس، عادي، مقبول، لكن/لاكن، «جيد لكن»). This supports Macro-F1 as the primary metric, the class-weight axis and word bigrams. Some neutral rows read as negative (rid 22), matching the label noise noted in `docs/PROBLEM_FRAMING.md`; neutral recall will be the weakest point to watch.
- **Negation carries sentiment signal**: a negation particle appears in 32.4% of negative and 27.1% of neutral reviews against 8.9% of positive ones (غير is among the top negative terms). Negations must survive preprocessing; stop-word removal is measured, not assumed.
- **Aspect cues are lexical but split by clitics and spelling**: الدونات/دونات/والدونات, قهوه/القهوه/قهوتهم, اسعار/الاسعار/اسعاره. Character n-grams (`char_wb`) or a word+character union are the expected way to share evidence across these forms; this is the first ablation axis.
- **Reviews are short**: median 5–8 words, and 19.8% have at most 3 words. Negative and neutral reviews are longer than positive ones (median 8 and 7 vs 5 words), so length is a candidate simple feature for sentiment (axis 5), to be kept only if it helps under the acceptance rule.
- **Emoji lean positive** (7.4% of positive vs 4.0% negative and 2.5% neutral reviews): emoji are kept as tokens in the symbols axis rather than stripped by default.
- **Aspect and sentiment are correlated**: service and prices are mostly negative, drinks, food and seating mostly positive. The two tasks stay separate models; the joint-correct rate is reported as a secondary metric.
- **Single-label limitation is real but limited**: about 9.8% of reviews mention two or more aspects by the cue estimate, and the labelled aspect is almost always one of them (97.2%). Multi-aspect reviews are therefore expected to explain only part of the aspect errors; error analysis in M4 will count them.

## Figures

- `reports/figures/dev_label_matrix.png`
- `reports/figures/dev_length_by_sentiment.png`
