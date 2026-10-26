"""Arabic demo page: type a cafe review, see its predicted sentiment and main aspect.

Run from the repository root (so .streamlit/config.toml applies):  streamlit run app/streamlit_app.py
URL options: ?example=<key> fills and analyses one of EXAMPLES (on load, and again whenever the value
changes in the same session, e.g. by back/forward); ?view=method opens the methodology tab.

A result is shown only for text that was submitted («حلّل المراجعة» or an example) and has not been
edited since; editing the text hides the old result until it is submitted again.

The page uses version 2 when its model card and saved pipelines are present, otherwise version 1
(decided once per session). It reaches the models only through ``get_card`` and ``run_prediction``,
applies no text transformation of its own and never shows a model score: certainty is shown as words,
and every number in the methodology tab is read from the model cards.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from arabic_reviews import config  # noqa: E402
from arabic_reviews.predict import (  # noqa: E402
    InvalidReview, MAX_CHARS, Prediction, TaskResult, available_versions, model_card, predict, thresholds_for)

APP_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- model versions
# One adapter per call; the page itself only passes a version name around.

VERSIONS = {"v1": "الإصدار 1", "v2": "الإصدار 2"}
BUILD_COMMANDS = {"v1": "python -m arabic_reviews final --check", "v2": "python -m arabic_reviews v2-build"}


def page_version() -> str:
    """Version 2 when its card and saved pipelines are present, otherwise version 1; fixed for the session."""
    if "version" not in st.session_state:
        st.session_state["version"] = "v2" if "v2" in available_versions() else "v1"
    return st.session_state["version"]


def get_card(version: str) -> dict:
    """The model card of one version."""
    if version not in VERSIONS:
        raise ValueError(f"unknown model version: {version}")
    return model_card(version)


def run_prediction(text: str, version: str) -> Prediction:
    """Score one review; raises InvalidReview or FileNotFoundError exactly like ``predict``."""
    if version not in VERSIONS:
        raise ValueError(f"unknown model version: {version}")
    return predict(text, version=version)


# --------------------------------------------------------------------------- page text


class Example(NamedTuple):
    button: str
    text: str


# Hand-written examples (not taken from the dataset). The keys are the values of ?example=.
EXAMPLES = {
    "coffee": Example("قهوة ممتازة", "القهوة ممتازة والطعم رائع"),
    "service": Example("خدمة بطيئة", "الخدمة بطيئة والموظف ما رد علينا"),
    "prices": Example("أسعار مرتفعة", "الأسعار غالية مقارنة بالكمية"),
    "seating": Example("جلسة هادئة", "الجلسة هادئة والمكان مريح"),
    "negation": Example("عبارة منفية", "ما عجبني الكيك"),
    "mixed": Example("رأي مختلط", "القهوة لذيذة لكن الأسعار مرتفعة"),
    "descriptive": Example("نص وصفي", "الفرع في شارع التحلية"),
}

NOTES = {
    "long_text": "المراجعة أطول من معظم مراجعات التدريب؛ قد تقل الدقة.",
    "very_short": "المراجعة قصيرة جدًا وقد لا تحمل سياقًا كافيًا.",
    "library_version": "إصدار المكتبة يختلف عن إصدار التدريب؛ أعد بناء النماذج.",
}
MISSING_MODELS = "النماذج المحفوظة غير موجودة. شغّل: `{command}`"

# Certainty words: they describe the model's own confidence, not the chance of being right.
CERTAINTY = {
    "high": ("ثقة النموذج: مرتفعة", "blue"),
    "medium": ("ثقة النموذج: متوسطة", "gray"),
    "low": ("ثقة النموذج: منخفضة — اقرأ النص بنفسك", "orange"),
}
LEVEL_NAMES = {"high": "مرتفعة", "medium": "متوسطة", "low": "منخفضة"}

# Display spelling of the model's labels (the labels themselves are unchanged).
LABEL_TEXT = {"ايجابي": "إيجابي", "الاسعار": "الأسعار"}
# Colour and Material Symbols icon name per label; the word itself always carries the meaning.
SENTIMENT_STYLE = {
    "ايجابي": ("green", "sentiment_satisfied"),
    "سلبي": ("red", "sentiment_dissatisfied"),
    "محايد": ("gray", "sentiment_neutral"),
}
ASPECT_ICONS = {
    "المأكولات": "bakery_dining",
    "المشروبات": "local_cafe",
    "الخدمة": "room_service",
    "الجلسات": "chair",
    "الاسعار": "payments",
}
NEUTRAL_MEANING = "«محايد» في هذه البيانات يعني غالبًا رأيًا مختلطًا أو فاترًا."

LIMITS = (
    "يحدد النموذج **جانبًا رئيسيًا واحدًا** لكل مراجعة، حتى لو ذكرت أكثر من جانب.",
    "المشاعر تقدير للمراجعة **كلها**، لا لكل جانب على حدة.",
    "النص الوصفي الخالي من الرأي (مثل عنوان الفرع) قد يأخذ وسمًا إيجابيًا أو سلبيًا غير مقصود، "
    "لأن مراجعات التدريب كلها تقريبًا تحمل رأيًا.",
)

# Plain tab names: an icon in a tab label would become part of the tab's spoken name.
TAB_ANALYSE = "تحليل مراجعة"
TAB_METHOD = "المنهجية والنتائج"


def label_text(label: str) -> str:
    return LABEL_TEXT.get(label, label)


def icon(name: str) -> str:
    """A decorative Material Symbols icon, hidden from screen readers (the text next to it says the same)."""
    return f'<span class="decor-icon" aria-hidden="true">{name}</span>' if name else ""


def bullets(lines) -> str:
    return "\n".join(f"- {line}" for line in lines)


def ltr(text: str) -> str:
    """Isolate a left-to-right term (Latin name, identifier) so it keeps its place in an Arabic sentence."""
    return f"⁦{text}⁩"


def f3(x: float) -> str:
    return f"{x:.3f}"


def pct(x: float) -> str:
    return f"{x:.0%}"


# --------------------------------------------------------------------------- state and callbacks


# "review" is the text area. "submitted" is the text of the last analysis request (None before the first):
# the result follows it, not the text area. "example_param" is the last ?example= value the page has
# applied; the callbacks below keep it equal to the URL, so a value that differs at the top of a run came
# from outside (a link, back/forward) and is applied once.

EDITED = "تغيّر النص بعد آخر تحليل. اضغط «حلّل المراجعة» لتحليل النص الحالي."
EMPTY = "ستظهر النتيجة هنا: مشاعر المراجعة وجانبها الرئيسي، ومدى ثقة النموذج في كلٍّ منهما."


def init_state() -> None:
    """Defaults, then any new ?example= value; runs before the text area is created."""
    st.session_state.setdefault("review", "")
    st.session_state.setdefault("submitted", None)
    st.session_state.setdefault("example_param", None)
    key = st.query_params.get("example")
    if key == st.session_state["example_param"]:
        return
    if key in EXAMPLES:
        st.session_state["review"] = st.session_state["submitted"] = EXAMPLES[key].text
        st.session_state["example_param"] = key
    else:
        if key is not None:
            st.query_params.pop("example")  # unknown example: keep the text, clean the URL
        st.session_state["example_param"] = None


def use_example(key: str) -> None:
    """An example chip fills the text area and submits it."""
    st.session_state["review"] = st.session_state["submitted"] = EXAMPLES[key].text
    st.session_state["example_param"] = key
    st.query_params["example"] = key


def submit_review() -> None:
    st.session_state["submitted"] = st.session_state["review"]


def clear_review() -> None:
    st.session_state["review"] = ""
    st.session_state["submitted"] = None
    st.session_state["example_param"] = None
    st.query_params.pop("example", None)


def review_edited() -> None:
    """Text edited by hand: the URL no longer names an example unless the text still is that example."""
    key = st.query_params.get("example")
    if key and (key not in EXAMPLES or EXAMPLES[key].text != st.session_state["review"]):
        st.query_params.pop("example", None)
        st.session_state["example_param"] = None


# --------------------------------------------------------------------------- result


def no_confident_level(spec: dict, label: str) -> bool:
    """True when the card gives this predicted label neither a high nor a medium level (version 2)."""
    thresholds = thresholds_for(spec, label)
    return thresholds.get("high") is None and thresholds.get("medium") is None


def render_task(title: str, task: TaskResult, kind: str, spec: dict) -> None:
    """One prediction: the label in large words, then the certainty in words (never a score)."""
    with st.container(key=f"result-{kind}", gap="small"):
        st.subheader(title, anchor=False)
        if kind == "sentiment":
            color, icon_name = SENTIMENT_STYLE.get(task.label, ("gray", ""))
        else:
            color, icon_name = "primary", ASPECT_ICONS.get(task.label, "")
        with st.container(key=f"label-{kind}"):
            st.markdown(f":{color}[{icon(icon_name)} {label_text(task.label)}]", unsafe_allow_html=True)
        words, badge_color = CERTAINTY[task.level]
        st.badge(words, color=badge_color)
        if task.second_label:
            st.caption(f"وقد تكون: **{label_text(task.second_label)}**")
        if no_confident_level(spec, task.label):
            st.caption(f"لا تُعرض ثقة مرتفعة أو متوسطة لتنبؤ «{label_text(task.label)}»: "
                       "لم تكن هذه التنبؤات موثوقة بما يكفي على بيانات التطوير.")
        if kind == "sentiment" and task.label == "محايد":
            st.caption(NEUTRAL_MEANING)


def render_result(result: Prediction, card: dict) -> None:
    tasks = card["tasks"]
    with st.container(border=True, key="result"):
        st.header("النتيجة", anchor=False)
        sentiment_col, aspect_col = st.columns(2, gap="medium")
        with sentiment_col:
            render_task("المشاعر", result.sentiment, "sentiment", tasks["sentiment"])
        with aspect_col:
            render_task("الجانب الرئيسي", result.aspect, "aspect", tasks["aspect"])
        st.caption(f"{icon('info')} النتيجة تقدير آلي من نموذج إحصائي، وقد تكون خاطئة.", unsafe_allow_html=True)
    for code in result.notes:
        alert = st.warning if code == "library_version" else st.info
        alert(NOTES.get(code, code), icon=":material/info:")


def render_limits() -> None:
    with st.container(key="limits", gap="small"):
        st.markdown("**حدود يجب معرفتها**")
        st.caption(bullets(LIMITS))


def render_analysis(version: str) -> None:
    st.text_area(
        "اكتب مراجعة عن مقهى",
        key="review",
        max_chars=MAX_CHARS,
        height=120,
        placeholder="اكتب رأيك في القهوة أو الأكل أو الخدمة أو الجلسات أو الأسعار",
        on_change=review_edited,
    )
    with st.container(horizontal=True, gap="small", key="actions"):
        st.button("حلّل المراجعة", type="primary", icon=":material/analytics:", key="analyse",
                  on_click=submit_review)
        st.button("مسح", type="tertiary", icon=":material/close:", key="clear", on_click=clear_review,
                  disabled=not st.session_state["review"])
    with st.container(key="examples", gap="small"):
        st.caption("أو جرّب مثالًا:")
        # The chips wrap into as many rows as the width needs, so every example stays visible.
        with st.container(horizontal=True, wrap=True, gap="small", key="example-row"):
            for key, example in EXAMPLES.items():
                st.button(example.button, key=f"example-{key}", on_click=use_example, args=(key,))

    review, submitted = st.session_state["review"], st.session_state["submitted"]
    if submitted is None or submitted != review:
        # Nothing submitted yet, or the text changed since: no result (an old one would describe other text).
        edited = submitted is not None and review.strip()
        with st.container(border=True, key="result-empty"):
            st.caption(f"{icon('edit_note')} {EDITED if edited else EMPTY}", unsafe_allow_html=True)
    else:
        try:
            with st.spinner("جارٍ التحليل…"):
                result = run_prediction(submitted, version)
        except InvalidReview as err:
            st.warning(err.message, icon=":material/edit_note:")
        except FileNotFoundError:
            st.error(MISSING_MODELS.format(command=BUILD_COMMANDS[version]), icon=":material/error:")
        else:
            render_result(result, get_card(version))
    render_limits()


# --------------------------------------------------------------------------- methodology


def _rows_for_levels(levels: list[dict]) -> dict:
    """Level table: share of test reviews in each level and how many of them were right."""
    total = sum(row.get("n", 0) for row in levels) or 1
    rows = {"الثقة": [], "الحصة": [], "الصحيح منها": [], "العدد": []}
    for row in levels:
        rows["الثقة"].append(LEVEL_NAMES.get(row["level"], row["level"]))
        rows["الحصة"].append(pct(row.get("share", row.get("n", 0) / total)))
        rows["الصحيح منها"].append(pct(row["accuracy"]))
        rows["العدد"].append(str(row.get("n", "—")))
    return rows


def _between(interval: dict | None) -> str:
    return f"بين {f3(interval['low'])} و{f3(interval['high'])}" if interval else ""


def _baseline_sentence(decision: dict | None) -> str:
    """The card's pass/fail comparison with the reference and dummy baselines, as one sentence."""
    if not decision:
        return ""
    parts = []
    if decision.get("reference") == "above":
        parts.append(f"أعلى من النموذج المرجعي ({ltr('TF-IDF')} للكلمات مع انحدار لوجستي)")
    elif decision.get("reference"):
        parts.append("ليس أعلى من النموذج المرجعي")
    dummies = decision.get("beats_dummies", {})
    if dummies:
        beaten = "النموذجين البسيطين" if len(dummies) == 2 else "كل النماذج البسيطة"
        parts.append(f"يتفوق على {beaten} ({ltr('Dummy')})" if all(dummies.values())
                     else f"لا يتفوق على كل النماذج البسيطة ({ltr('Dummy')})")
    return "، و".join(parts) + "." if parts else ""


def _measured(card: dict) -> str:
    """How often the held-out test was read, from the card's test record."""
    run = card["test_run"]["run"]
    return "مرة واحدة" if run == 1 else f"في التشغيل المسجّل رقم {ltr(str(run))}"


def render_methodology(version_label: str, card: dict) -> None:
    """Methodology and frozen results of one model version; every number comes from ``card``."""
    tasks = card.get("tasks", {})
    sentiment, aspect = tasks.get("sentiment", {}), tasks.get("aspect", {})
    test_rows = card.get("test_run", {}).get("test_rows")

    st.header(f"{version_label}: نتائج مجمَّدة من اختبار محجوز", anchor=False)
    st.caption(
        f"قيست هذه النتائج {_measured(card)} على مراجعات محجوزة لم تُستخدم في التدريب ولا في أي اختيار، "
        "بعد تجميد النموذج والعتبات."
    )

    st.subheader("المهمة والوسوم", anchor=False)
    st.markdown(bullets((
        "مهمتا تصنيف مستقلتان على نص المراجعة نفسه، ولكل مراجعة وسم واحد في كل مهمة.",
        "**الجانب الرئيسي:** " + "، ".join(label_text(x) for x in aspect.get("labels", [])) + ". "
        "«الخدمة» تشمل في هذه البيانات المواقف والموقع وخدمة السيارات أيضًا.",
        "**المشاعر** للمراجعة كلها: " + "، ".join(label_text(x) for x in sentiment.get("labels", [])) + ".",
        "**«محايد»** في هذه البيانات يعني غالبًا رأيًا **مختلطًا** (إيجابي وسلبي معًا) أو **فاترًا**، "
        "لا نصًا خاليًا من الرأي. وهي أصغر الفئات وأصعبها.",
    )))
    st.caption(f"تعريف الوسوم بأمثلة حقيقية في {ltr('docs/PROBLEM_FRAMING.md')}")

    st.subheader("البيانات والترخيص", anchor=False)
    source_surname = config.SOURCE["creator"].split()[-1]
    source_year = config.SOURCE["publication_date"][:4]
    data_lines = [f"مراجعات {ltr('Google Maps')} لمقاهي الرياض، بلهجة سعودية غالبًا "
                  f"({ltr(f'{source_surname}, {source_year}')})."]
    source = card.get("source")
    if source:
        data_lines.append(f"المصدر: [صفحة البيانات على {ltr('Zenodo')}]({source['record']})، "
                          f"والمعرّف `{source['doi']}`")
        data_lines.append(f"الترخيص: `{source['license']}`، ولا تُعاد استضافة البيانات في المستودع.")
    trained = card.get("trained_on", {}).get("rows")
    if trained and test_rows:
        data_lines.append(f"التدريب على {trained} مراجعة تطوير، والاختبار المحجوز {test_rows} مراجعة.")
    st.markdown(bullets(data_lines))

    st.subheader("النتائج على الاختبار المحجوز", anchor=False)
    st.caption(f"المقياس الأساسي {ltr('Macro-F1')} لأن الفئات غير متوازنة؛ أعلى قيمة له 1.")
    metric_cols = st.columns(3)
    for col, title, spec in ((metric_cols[0], f"المشاعر · {ltr('Macro-F1')}", sentiment),
                             (metric_cols[1], f"الجانب · {ltr('Macro-F1')}", aspect)):
        test = spec.get("test", {})
        if "macro_f1" not in test:
            continue
        with col.container(border=True, height="stretch"):
            st.metric(title, f3(test["macro_f1"]))
            if test.get("interval"):
                st.caption(f"فاصل الثقة: {_between(test['interval'])}")
            if spec.get("expected_range"):
                st.caption(f"المتوقع مسبقًا: {_between(spec['expected_range'])}")
    if "joint_test_accuracy" in card:
        with metric_cols[2].container(border=True, height="stretch"):
            st.metric("التطابق المشترك", f3(card["joint_test_accuracy"]))
            st.caption("نسبة المراجعات التي صحّ فيها الجانب والمشاعر معًا.")

    decisions = {name: _baseline_sentence(spec.get("test", {}).get("decision"))
                 for name, spec in (("المشاعر", sentiment), ("الجانب", aspect))}
    decisions = {name: text for name, text in decisions.items() if text}
    if len(decisions) == 2 and len(set(decisions.values())) == 1:
        decisions = {"المشاعر والجانب": "كلاهما " + next(iter(decisions.values()))}
    if decisions:
        st.markdown("**مقارنة بخطوط الأساس**\n\n" + bullets(f"**{name}:** {text}" for name, text in decisions.items()))

    f1_cols = st.columns(2)
    for col, name, spec in ((f1_cols[0], "المشاعر", sentiment), (f1_cols[1], "الجانب", aspect)):
        per_class = spec.get("test", {}).get("per_class_f1")
        if per_class:
            with col:
                st.markdown(f"**{ltr('F1')} لكل فئة · {name}**")
                st.table({"الفئة": [label_text(k) for k in per_class], "F1": [f3(v) for v in per_class.values()]},
                         hide_index=True, alt=f"F1 لكل فئة على الاختبار المحجوز · {name}")

    st.subheader("ثقة النموذج بقاعدة الإصدار 1", anchor=False)
    st.markdown(
        "مستوى الثقة وصف لنطاق أعلى درجة يعطيها النموذج لتنبؤه، **وليس احتمالًا معايرًا**. "
        "في قاعدة الإصدار 1 زوج عتبات واحد لكل مهمة، يسري على كل الفئات. "
        f"الجدولان يبيّنان ما لوحظ بهذه القاعدة {_measured(card)} على الاختبار المحجوز: حصة المراجعات في كل مستوى، "
        "ونسبة التنبؤات الصحيحة فيه. هذه تكرارات رُصدت على بيانات سابقة، **وليست ضمانًا** لصحة أي تنبؤ جديد."
    )
    level_cols = st.columns(2)
    for col, title, spec in ((level_cols[0], "المشاعر", sentiment), (level_cols[1], "الجانب الرئيسي", aspect)):
        levels = spec.get("certainty", {}).get("test_levels")
        if levels:
            with col:
                st.markdown(f"**{title}: قاعدة الإصدار 1 (زوج عتبات واحد)**")
                st.table(_rows_for_levels(levels), hide_index=True,
                         alt=f"مستويات الثقة بقاعدة الإصدار 1 على الاختبار المحجوز · {title}")


# --------------------------------------------------------------------------- version 2

CANDIDATES = {"V1": "الإصدار 1", "H1_phrases": "ثنائيات الكلمات", "H2_cues": "دلائل القطبية", "H3_both": "الاثنان معًا"}


def signed(x: float) -> str:
    """A signed number with a true minus sign, kept left-to-right inside Arabic text."""
    return ltr(f"{x:+.3f}".replace("-", "−"))


def _level_changes(v1_spec: dict, v2_spec: dict) -> list[str]:
    """What the per-label rule of version 2 changes against the single version-1 pair, in words."""
    certainty = v2_spec.get("certainty", {})
    old = v1_spec.get("certainty", {}).get("thresholds") or certainty.get("v1_thresholds")
    if not old:
        return []
    lines = []
    for label, new in certainty.get("thresholds_by_class", {}).items():
        name = f"«{label_text(label)}»"
        high, medium = new.get("high"), new.get("medium")
        if high is None and medium is None:
            lines.append(f"لا يُعرض تنبؤ {name} بثقة مرتفعة أو متوسطة.")
            continue
        for level, value, word in (("high", high, "مرتفعة"), ("medium", medium, "متوسطة")):
            if value is None:
                lines.append(f"لا تُعرض ثقة «{word}» لتنبؤ {name}.")
            elif value > old[level]:
                lines.append(f"يحتاج تنبؤ {name} درجة أعلى من الإصدار 1 ليُعرض بثقة {word}.")
            elif value < old[level]:
                lines.append(f"يُعرض تنبؤ {name} بثقة {word} من درجة أدنى من الإصدار 1.")
    return lines


def render_overview(v1_card: dict, v2_card: dict) -> None:
    """What the page runs when version 2 is in use: version-1 classifiers, version-2 certainty words."""
    tasks1, tasks2 = v1_card.get("tasks", {}), v2_card.get("tasks", {})
    with st.container(border=True, key="overview"):
        st.header("ما الذي يعمل في هذه الصفحة؟", anchor=False)
        if tasks2 and all(spec.get("same_as_v1") for spec in tasks2.values()):
            first = (f"**المصنِّفان** هما مصنِّفا الإصدار 1 دون أي تغيير، وقد قِيسا {_measured(v1_card)} على "
                     "اختبار محجوز (القسم التالي).")
        else:
            first = "**المصنِّفان** من الإصدار 2، ولم يُقاسا على الاختبار المحجوز."
        changes = []
        for task in ("sentiment", "aspect"):
            changes += _level_changes(tasks1.get(task, {}), tasks2.get(task, {}))
        second = ("**الإصدار 2 يغيّر عرض الثقة فقط:** لكل فئة متنبأ بها مستوياتها، استُخرجت وفُحصت على "
                  "بيانات التطوير وحدها. يمكن أن يُرفع مستوى أو يُحذف مقارنة بالإصدار 1، ولا يمكن أن يتسع.")
        second += "".join(f"\n  - {line}" for line in changes)
        st.markdown(bullets((first, second, "لم يُقس أي شيء في الإصدار 2 على الاختبار المحجوز.")))


def _v2_level_rows(spec: dict) -> dict | None:
    """Per predicted label and level on the outer development folds: version-2 rule, then version-1 rule."""
    certainty = spec.get("certainty", {})
    outer = certainty.get("outer_levels")
    if not outer:
        return None
    by_class = certainty.get("thresholds_by_class", {})
    old = {(r["label"], r["level"]): r for r in certainty.get("v1_rule_outer_levels", [])}
    # Most telling columns first (a phone shows them without scrolling sideways).
    rows = {"الفئة": [], "الثقة": [], "الصحيح منها": [], "بقاعدة الإصدار 1": [], "فاصل الثقة": [],
            "الحصة": [], "الصفوف": []}
    if not old:
        del rows["بقاعدة الإصدار 1"]
    for r in outer:
        shown = r["level"] == "low" or by_class.get(r["label"], {}).get(r["level"]) is not None
        rows["الفئة"].append(label_text(r["label"]))
        rows["الثقة"].append(LEVEL_NAMES.get(r["level"], r["level"]))
        if shown and r.get("rows"):
            interval = r.get("interval")
            rows["الحصة"].append(pct(r["share"]))
            rows["الصحيح منها"].append(pct(r["accuracy"]))
            rows["فاصل الثقة"].append(ltr(f"{pct(interval[0])}–{pct(interval[1])}") if interval else "—")
        else:
            rows["الحصة"].append("—")
            rows["الصحيح منها"].append("لا يُعرض" if not shown else "—")
            rows["فاصل الثقة"].append("—")
        rows["الصفوف"].append(str(r.get("rows", 0)))
        if old:
            before = old.get((r["label"], r["level"]))
            rows["بقاعدة الإصدار 1"].append(
                f"{pct(before['accuracy'])} ({before['rows']})" if before and before.get("rows") else "—")
    return rows


def _removed_examples(spec: dict) -> list[str]:
    """Labels left with no high or medium level, and how right those levels were under the version-1 rule."""
    certainty = spec.get("certainty", {})
    old = [r for r in certainty.get("v1_rule_outer_levels", []) if r["level"] != "low" and r.get("rows")]
    lines = []
    for label in certainty.get("thresholds_by_class", {}):
        if not no_confident_level(spec, label):
            continue
        parts = [f"«{LEVEL_NAMES[r['level']]}» (صحيح في {pct(r['accuracy'])})" for r in old if r["label"] == label]
        if parts:
            lines.append(f"بقاعدة الإصدار 1 كان تنبؤ «{label_text(label)}» يُعرض أحيانًا بثقة "
                         + " أو ".join(parts) + "؛ في الإصدار 2 لا يُعرض بأيٍّ منهما.")
    return lines


def render_v2_certainty(version_label: str, card: dict) -> None:
    """Per-label certainty of version 2 on development data; every number comes from ``card``."""
    tasks = card.get("tasks", {})
    st.header(f"{version_label}: الثقة حسب الفئة (بيانات التطوير فقط)", anchor=False)
    st.markdown(
        "لكل فئة متنبأ بها مستوياتها الخاصة، استُخرجت من بيانات التطوير وحدها. لا يبدأ أي مستوى تحت عتبته "
        "في الإصدار 1، فيمكن أن يُرفع أو يُحذف لفئة ولا يمكن أن يتسع؛ ويُحذف مستوى إذا لم تبلغ دقته الهدف "
        "أو لم تكفِ البيانات لإثباته. اختيرت هذه القاعدة بعد رؤية نتائج القاعدة المسجلة مسبقًا، التي كانت "
        "توسّع بعض المستويات، وهي الأكثر تحفظًا بينهما (القاعدتان في `reports/v2/results.md`)."
    )
    examples = [line for task in ("sentiment", "aspect") for line in _removed_examples(tasks.get(task, {}))]
    if examples:
        st.markdown(bullets(examples))
    for task, title in (("sentiment", "المشاعر"), ("aspect", "الجانب الرئيسي")):
        rows = _v2_level_rows(tasks.get(task, {}))
        if rows:
            with st.expander(f"{title}: المستويات حسب الفئة", key=f"v2-levels-{task}"):
                with st.container(key=f"scroll-hint-{task}"):
                    st.caption("اسحب الجدول جانبيًا لرؤية بقية الأعمدة.")
                st.table(rows, hide_index=True, alt=f"مستويات الثقة في الإصدار 2 على بيانات التطوير · {title}")
    st.caption(
        "تكرارات رُصدت على بيانات التطوير (الطيات الخارجية للتحقق المتقاطع المكرَّر، وتظهر كل مراجعة "
        "في كل تكرار)، "
        "وليست ضمانًا لصحة أي تنبؤ جديد؛ ولا يوجد اختبار مستقل للإصدار 2. «بقاعدة الإصدار 1»: نسبة الصحيح "
        "(وعدد الصفوف) بزوج عتبات الإصدار 1 على الطيات نفسها. الحصة: نسبة الخانة من كل التنبؤات؛ "
        "الصفوف: عدد التنبؤات."
    )


def render_experiment(card: dict) -> None:
    """The neutral-class experiment of version 2 and why it was or was not adopted, from ``card``."""
    experiment = card.get("experiment")
    if not experiment:
        return
    adopted = experiment.get("adopted")
    st.header("تجربة تحسين المحايد (" + ("اعتُمدت" if adopted else "لم تُعتمد") + ")", anchor=False)
    candidates = experiment.get("candidates", {})
    if candidates:
        st.table({
            "المرشح": [CANDIDATES.get(name, name) for name in candidates],
            "Macro-F1": [f3(c["selection_macro_f1"]) for c in candidates.values()],
            "F1 المحايد": [f3(c["selection_neutral_f1"]) for c in candidates.values()],
        }, hide_index=True, alt="مرشحو تجربة المحايد على طيات الاختيار")
        st.caption("على طيات الاختيار من بيانات التطوير.")
    text = []
    chosen = experiment.get("chosen_on_selection_folds")
    if chosen:
        text.append(f"اختارت القاعدة المعلنة «{CANDIDATES.get(chosen, chosen)}» على طيات الاختيار.")
    nested = card.get("nested_vs_v1", {})
    gain, neutral = nested.get("gain"), nested.get("neutral")
    if gain:
        text.append(f"لكن تقدير الإجراء كله بتحقق متقاطع متداخل أعطى مكسبًا في {ltr('Macro-F1')} قدره "
                    f"{signed(gain['point'])}، بفاصل ثقة بين {signed(gain['low'])} و{signed(gain['high'])}.")
    if neutral:
        text.append(f"وتغيّر {ltr('F1')} المحايد المقدَّر {signed(neutral['point'])}، بفاصل بين "
                    f"{signed(neutral['low'])} و{signed(neutral['high'])}.")
    if not adopted:
        reason = ("ولأن فاصل المكسب لا يقع كله فوق الصفر، كما يشترط المعيار المعلن مسبقًا،"
                  if gain and gain["low"] <= 0 else "ولأن شروط الاعتماد المعلنة مسبقًا لم تتحقق كلها،")
        text.append(f"{reason} لم يُعتمد نموذج مشاعر جديد، وبقي مصنِّف الإصدار 1.")
    text.append("الدليل كله من بيانات التطوير نفسها التي بُني عليها التصميم، فلا يُعدّ تأكيدًا مستقلًا.")
    st.markdown(" ".join(text))


# --------------------------------------------------------------------------- limits and details


def render_known_limits(version: str) -> None:
    st.header("حدود معروفة", anchor=False)
    lines = [
        "جانب رئيسي واحد لكل مراجعة؛ لا يحلل النموذج مشاعر كل جانب مذكور في النص.",
        f"النطاق مقاهي الرياض على {ltr('Google Maps')}؛ لم يُقس الأداء على مطاعم أو لهجات أخرى.",
        "السخرية والرأي الضمني لم يُقيَّما منفصلين، وقد يُصنَّفان خطأً.",
        "النفي لا يُلتقط إلا من تتابع الكلمات والحروف.",
        "النص الوصفي الخالي من الرأي قد يأخذ وسمًا إيجابيًا أو سلبيًا غير مقصود.",
        "نص من حروف عربية بلا معنى لا يُرفض دائمًا؛ ثقة النموذج المنخفضة هي الإشارة الوحيدة حينها.",
    ]
    if version == "v2":
        lines.append("تقديرات الإصدار 2 من بيانات التطوير وحدها؛ لا يوجد اختبار مستقل للإصدار 2.")
    st.markdown(bullets(lines))


def render_details(version: str, v1_card: dict, v2_card: dict | None) -> None:
    with st.expander("تفاصيل تقنية", icon=":material/tune:"):
        if version == "v2":
            lines = ["قاعدة الثقة المستخدمة في الصفحة: **الإصدار 2**، مستويات لكل فئة متنبأ بها."]
        else:
            lines = ["قاعدة الثقة المستخدمة في الصفحة: **الإصدار 1**، زوج عتبات واحد لكل مهمة. "
                     f"الإصدار 2 غير متاح هنا (بطاقته أو نماذجه غير موجودة)؛ لبنائه: `{BUILD_COMMANDS['v2']}`"]
        for name, task in (("المشاعر", "sentiment"), ("الجانب", "aspect")):
            cfg = v1_card.get("tasks", {}).get(task, {}).get("configuration")
            if cfg:
                params = "، ".join(f"`{k} = {v}`" for k, v in cfg.get("params", {}).items())
                lines.append(f"**{name}:** التمثيل `{cfg['representation']}`، النموذج `{cfg['model']}`، "
                             f"أوزان الفئات `{cfg['class_weight']}`" + (f"، {params}" if params else ""))
        if v1_card.get("fingerprint"):
            lines.append(f"بصمة البروتوكول المجمَّد للإصدار 1: `{v1_card['fingerprint'][:8]}`")
        if v2_card and v2_card.get("fingerprint"):
            lines.append(f"بصمة الإصدار 2: `{v2_card['fingerprint'][:8]}`")
        versions = (v2_card or v1_card).get("versions")
        if versions:
            lines.append("الإصدارات: " + "، ".join(f"`{k} {v}`" for k, v in versions.items()))
        lines.append("التقارير: `reports/final_test.md`" + ("، `reports/v2/results.md`" if v2_card else ""))
        st.markdown(bullets(lines))


def render_method_tab(version: str) -> None:
    try:
        v1_card = get_card("v1")
        v2_card = get_card("v2") if version == "v2" else None
    except FileNotFoundError:
        st.error("بطاقة النموذج غير موجودة في `models/`.", icon=":material/error:")
        return
    if v2_card:
        render_overview(v1_card, v2_card)
    render_methodology(VERSIONS["v1"], v1_card)
    if v2_card:
        render_v2_certainty(VERSIONS["v2"], v2_card)
        render_experiment(v2_card)
    render_known_limits(version)
    render_details(version, v1_card, v2_card)


# --------------------------------------------------------------------------- page

st.set_page_config(page_title="تحليل مراجعات المقاهي", page_icon=":material/local_cafe:", layout="centered")
st.html(APP_DIR / "style.css")
init_state()
version = page_version()

st.title("تحليل مراجعات المقاهي", anchor=False)
st.markdown("اكتب مراجعة عربية عن مقهى، ويقدّر النموذج **مشاعرها** و**جانبها الرئيسي**.")
st.caption("مشروع تعليمي، وليس نظامًا إنتاجيًا. النموذج مدرَّب على مراجعات مقاهي الرياض فقط.")

analyse_tab, method_tab = st.tabs(
    [TAB_ANALYSE, TAB_METHOD], default=TAB_METHOD if st.query_params.get("view") == "method" else None
)
with analyse_tab:
    render_analysis(version)
with method_tab:
    render_method_tab(version)

# Arabic page language for screen readers and hyphenation; the direction is set in style.css.
st.html("<script>document.documentElement.lang = 'ar';</script>", unsafe_allow_javascript=True)
