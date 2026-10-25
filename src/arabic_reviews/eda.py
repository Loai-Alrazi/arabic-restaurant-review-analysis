"""Exploratory analysis of the development partition only (reads ``split.load_dev()``).

Writes ``reports/eda.md`` (deterministic) and two figures. Figures use English
label names because matplotlib does not shape Arabic script.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

from . import config
from .split import load_dev
from .text import FULL, NEGATIONS, normalize

LABEL_EN = {
    "المأكولات": "food", "المشروبات": "drinks", "الخدمة": "service", "الجلسات": "seating", "الاسعار": "prices",
    "ايجابي": "positive", "سلبي": "negative", "محايد": "neutral",
}
# Rough stems per aspect, matched on FULL-normalised text, used only to estimate how
# often a review mentions more than one aspect. Not a feature and not a label source.
ASPECT_CUES = {
    "المأكولات": r"اكل|طعام|دونات|كروس|كرواس|كيك|كوكيز|ساندو?يش|سندو?يش|سانويش|حلويات|حلي?ا\b|فطور|تشيز|وجب|تراميسو",
    "المشروبات": r"قهو|قهاوي|كوفي|لاتيه|موكا|مشروب|شاي|عصير|كابتشينو|كابوتشينو|اسبريسو|ايس\b|برو\b|كراميل",
    "الخدمة": r"خدم|موظف|تعامل|طلب|انتظار|مواقف|استقبال|عامل|كاشير|سريع|بطيء|تاخير",
    "الجلسات": r"مكان|جلس|هادي|هادئ|اجواء|ديكور|نظاف|نظيف|زحم|مزدحم|ازعاج",
    "الاسعار": r"سعر|اسعار|غالي|رخيص|مبالغ|ريال|تكلف",
}
EMOJI_RE = re.compile("[\U0001f300-\U0001faff☀-➿]")
TOKEN_RE = r"(?u)\b\w+\b"


def _tokens(text: str) -> list[str]:
    return re.findall(TOKEN_RE, normalize(text, **FULL))


def _has_negation(text: str) -> bool:
    return any(t in NEGATIONS for t in _tokens(text))


def _md_table(frame: pd.DataFrame, fmt: str = "{}") -> list[str]:
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join([frame.index.name or ""] + cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, row in frame.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [fmt.format(v) for v in row]) + " |")
    return lines


def distinctive_ngrams(texts: pd.Series, labels: pd.Series, k: int = 8) -> dict[str, list[str]]:
    """Top ``k`` word 1-2-grams per class by one-vs-rest chi-squared, positively associated only."""
    vec = CountVectorizer(token_pattern=TOKEN_RE, ngram_range=(1, 2), min_df=3, binary=True,
                          preprocessor=lambda t: normalize(t, **FULL))
    X = vec.fit_transform(texts)
    terms = vec.get_feature_names_out()
    out = {}
    for label in sorted(labels.unique()):
        y = (labels == label).to_numpy()
        scores, _ = chi2(X, y)
        rate_in = np.asarray(X[y].mean(axis=0)).ravel()
        rate_out = np.asarray(X[~y].mean(axis=0)).ravel()
        order = sorted((i for i in range(len(terms)) if rate_in[i] > rate_out[i]), key=lambda i: (-scores[i], terms[i]))
        out[label] = [terms[i] for i in order[:k]]
    return out


def multi_aspect_estimate(dev: pd.DataFrame) -> dict:
    norm = dev.text.map(lambda t: normalize(t, **FULL))
    hits = pd.DataFrame({a: norm.str.contains(p, regex=True) for a, p in ASPECT_CUES.items()})
    n_hit = hits.sum(axis=1)
    label_hit = hits.to_numpy()[np.arange(len(dev)), [config.ASPECTS.index(a) for a in dev.aspect]]
    single = n_hit == 1
    return {
        "rows": len(dev),
        "cue_counts": {k: int(v) for k, v in n_hit.clip(upper=3).value_counts().sort_index().items()},
        "single_cue_agrees_with_label_pct": float(label_hit[single].mean() * 100),
        "multi_cue_label_among_cues_pct": float(label_hit[n_hit >= 2].mean() * 100),
        "multi_cue_pct": float((n_hit >= 2).mean() * 100),
    }


def _figures(dev: pd.DataFrame) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    joint = pd.crosstab(dev.aspect, dev.sentiment).loc[list(config.ASPECTS), list(config.SENTIMENTS)]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.imshow(joint.to_numpy(), cmap="Blues")
    ax.set_xticks(range(3), [LABEL_EN[s] for s in joint.columns])
    ax.set_yticks(range(5), [LABEL_EN[a] for a in joint.index])
    for (i, j), v in np.ndenumerate(joint.to_numpy()):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > joint.to_numpy().max() / 2 else "black")
    ax.set_title(f"Development partition: aspect x sentiment (n={len(dev)})")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "dev_label_matrix.png", dpi=120, metadata={"Software": None})
    plt.close(fig)

    words = dev.text.str.split().str.len()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    data = [words[dev.sentiment == s].clip(upper=60) for s in config.SENTIMENTS]
    ax.boxplot(data, tick_labels=[LABEL_EN[s] for s in config.SENTIMENTS], showfliers=True)
    ax.set_ylabel("words per review (clipped at 60)")
    ax.set_title("Development partition: review length by sentiment")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "dev_length_by_sentiment.png", dpi=120, metadata={"Software": None})
    plt.close(fig)
    return ["figures/dev_label_matrix.png", "figures/dev_length_by_sentiment.png"]


def run_eda(figures: bool = True) -> str:
    dev = load_dev()
    dev = dev.assign(chars=dev.text.str.len(), words=dev.text.str.split().str.len(),
                     negation=dev.text.map(_has_negation), emoji=dev.text.str.contains(EMOJI_RE))
    out = [
        "# Exploratory data analysis — development partition only",
        "",
        "Written by `python -m arabic_reviews eda` from `split.load_dev()`; the held-out test partition is"
        " not read. Counts are rows of the dev partition.",
        "",
        f"Rows: **{len(dev)}**, near-duplicate groups: {dev.group_id.nunique()}.",
        "",
        "## 1. Labels",
        "",
    ]
    joint = pd.crosstab(dev.aspect, dev.sentiment, margins=True, margins_name="total")
    joint = joint.loc[[*config.ASPECTS, "total"], [*config.SENTIMENTS, "total"]]
    joint.index.name = "aspect \\ sentiment"
    out += _md_table(joint)
    neutral = (dev.sentiment == "محايد").groupby(dev.aspect).mean().loc[list(config.ASPECTS)] * 100
    out += ["", "Neutral share per aspect: " + ", ".join(f"{a} {v:.1f}%" for a, v in neutral.items()) + ".",
            "", "## 2. Text length", ""]
    for task, labels in (("sentiment", config.SENTIMENTS), ("aspect", config.ASPECTS)):
        stats = dev.groupby(task)[["chars", "words"]].describe(percentiles=[0.5, 0.9])
        table = pd.DataFrame({
            "rows": stats[("words", "count")].astype(int),
            "chars median": stats[("chars", "50%")], "chars p90": stats[("chars", "90%")],
            "words median": stats[("words", "50%")], "words p90": stats[("words", "90%")],
            "words max": stats[("words", "max")].astype(int),
        }).loc[list(labels)]
        table.index.name = task
        out += _md_table(table, "{:g}") + [""]
    out += [f"Reviews with at most 3 words: {(dev.words <= 3).mean() * 100:.1f}% of dev.", "",
            "## 3. Negation and emoji by sentiment", ""]
    rates = dev.groupby("sentiment")[["negation", "emoji"]].mean().loc[list(config.SENTIMENTS)] * 100
    rates.columns = ["% with a negation particle", "% with an emoji"]
    out += _md_table(rates, "{:.1f}")
    out += ["", f"Negation particles: `text.NEGATIONS` ({len(NEGATIONS)} forms), matched as whole tokens after"
            " full normalisation.", "", "## 4. Most distinctive word 1–2-grams per class (chi², one-vs-rest)", "",
            "Fitted on the dev partition for description only; model features are fitted inside each CV fold.", ""]
    for task in ("sentiment", "aspect"):
        grams = distinctive_ngrams(dev.text, dev[task])
        labels = config.SENTIMENTS if task == "sentiment" else config.ASPECTS
        out += [f"| {task} | top n-grams |", "|---|---|"]
        out += [f"| {lab} | {'، '.join(grams[lab])} |" for lab in labels] + [""]
    m = multi_aspect_estimate(dev)
    out += [
        "## 5. How often does a review mention more than one aspect?",
        "",
        "Estimated with a small hand-written list of word stems per aspect (`eda.ASPECT_CUES`), matched on"
        " normalised text. This is a rough indicator for the single-label limitation, not a measurement.",
        "",
        "| aspects with a cue | 0 | 1 | 2 | 3+ |", "|---|---|---|---|---|",
        "| reviews | " + " | ".join(str(m["cue_counts"].get(k, 0)) for k in range(4)) + " |",
        "",
        f"- **{m['multi_cue_pct']:.1f}%** of dev reviews contain cues for two or more aspects; for these, the"
        f" labelled aspect is among the cued ones in {m['multi_cue_label_among_cues_pct']:.1f}% of cases.",
        f"- Where exactly one aspect is cued, it agrees with the label in"
        f" {m['single_cue_agrees_with_label_pct']:.1f}% of cases (a sanity check of the cue list).",
        "",
        "## 6. Examples (first two dev rows per aspect, by rid)",
        "",
        "| rid | aspect | sentiment | text |", "|---|---|---|---|",
    ]
    for aspect in config.ASPECTS:
        for r in dev[dev.aspect == aspect].sort_values("rid").head(2).itertuples():
            out.append(f"| {r.rid} | {r.aspect} | {r.sentiment} | {r.text.replace('|', '/').replace(chr(10), ' ')} |")
    out += ["", "## 7. Observations that inform M3", ""]
    out += OBSERVATIONS
    if figures:
        out += ["", "## Figures", ""] + [f"- `reports/{p}`" for p in _figures(dev)]
    text = "\n".join(out) + "\n"
    config.EDA_REPORT.write_text(text, encoding="utf-8", newline="\n")
    return text


# Written after reading the tables above; kept in code so the report stays reproducible.
OBSERVATIONS = [
    "- **Neutral is rare (9.2%) and lexically hedged**: its distinctive terms are concessive or lukewarm"
    " words (جيد، بس، عادي، مقبول، لكن/لاكن، «جيد لكن»). This supports Macro-F1 as the primary metric, the"
    " class-weight axis and word bigrams. Some neutral rows read as negative (rid 22), matching the label"
    " noise noted in `docs/PROBLEM_FRAMING.md`; neutral recall will be the weakest point to watch.",
    "- **Negation carries sentiment signal**: a negation particle appears in 32.4% of negative and 27.1% of"
    " neutral reviews against 8.9% of positive ones (غير is among the top negative terms). Negations must"
    " survive preprocessing; stop-word removal is measured, not assumed.",
    "- **Aspect cues are lexical but split by clitics and spelling**: الدونات/دونات/والدونات, قهوه/القهوه/قهوتهم,"
    " اسعار/الاسعار/اسعاره. Character n-grams (`char_wb`) or a word+character union are the expected way to"
    " share evidence across these forms; this is the first ablation axis.",
    "- **Reviews are short**: median 5–8 words, and 19.8% have at most 3 words. Negative and neutral reviews"
    " are longer than positive ones (median 8 and 7 vs 5 words), so length is a candidate simple feature"
    " for sentiment (axis 5), to be kept only if it helps under the acceptance rule.",
    "- **Emoji lean positive** (7.4% of positive vs 4.0% negative and 2.5% neutral reviews): emoji are kept"
    " as tokens in the symbols axis rather than stripped by default.",
    "- **Aspect and sentiment are correlated**: service and prices are mostly negative, drinks, food and"
    " seating mostly positive. The two tasks stay separate models; the joint-correct rate is reported as a"
    " secondary metric.",
    "- **Single-label limitation is real but limited**: about 9.8% of reviews mention two or more aspects by"
    " the cue estimate, and the labelled aspect is almost always one of them (97.2%). Multi-aspect reviews"
    " are therefore expected to explain only part of the aspect errors; error analysis in M4 will count them.",
]
