"""Arabic demo page: type a cafe review, see its predicted sentiment and main aspect.

Run from the repository root (so .streamlit/config.toml applies):  streamlit run app/streamlit_app.py
URL options: ?example=<key> fills one of EXAMPLES so its result shows on load; ?view=method opens
the methodology tab.

The page reaches the models only through ``get_card`` and ``run_prediction``. It applies no text
transformation of its own and never shows a model score: certainty is shown as words, and every
number in the methodology tab is read from the model card.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from arabic_reviews.predict import InvalidReview, MAX_CHARS, Prediction, TaskResult, model_card, predict  # noqa: E402

APP_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- model versions
# One adapter per call, so a second version can be added here without touching the page.

VERSIONS = {"v1": "الإصدار 1"}
DEFAULT_VERSION = "v1"


def get_card(version: str = DEFAULT_VERSION) -> dict:
    """The model card of one version."""
    if version == "v1":
        return model_card()
    raise ValueError(f"unknown model version: {version}")


def run_prediction(text: str, version: str = DEFAULT_VERSION) -> Prediction:
    """Score one review; raises InvalidReview or FileNotFoundError exactly like ``predict``."""
    if version == "v1":
        return predict(text)
    raise ValueError(f"unknown model version: {version}")


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
    "very_short": "المراجعة قصيرة جدًا؛ النتيجة أقل ثباتًا.",
    "library_version": "إصدار المكتبة يختلف عن إصدار التدريب؛ أعد بناء النماذج.",
}
MISSING_MODELS = "النماذج المحفوظة غير موجودة. شغّل: `python -m arabic_reviews final --check`"

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


def init_state() -> None:
    if "review" not in st.session_state:
        key = st.query_params.get("example")
        st.session_state["review"] = EXAMPLES[key].text if key in EXAMPLES else ""


def use_example(key: str) -> None:
    st.session_state["review"] = EXAMPLES[key].text
    st.query_params["example"] = key


def clear_review() -> None:
    st.session_state["review"] = ""
    st.query_params.pop("example", None)


def review_edited() -> None:
    key = st.query_params.get("example")
    if key and (key not in EXAMPLES or EXAMPLES[key].text != st.session_state["review"]):
        st.query_params.pop("example", None)


# --------------------------------------------------------------------------- result


def render_task(title: str, task: TaskResult, kind: str) -> None:
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
        if kind == "sentiment" and task.label == "محايد":
            st.caption(NEUTRAL_MEANING)


def render_result(result: Prediction) -> None:
    with st.container(border=True, key="result"):
        st.header("النتيجة", anchor=False)
        sentiment_col, aspect_col = st.columns(2, gap="medium")
        with sentiment_col:
            render_task("المشاعر", result.sentiment, "sentiment")
        with aspect_col:
            render_task("الجانب الرئيسي", result.aspect, "aspect")
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
        st.button("حلّل المراجعة", type="primary", icon=":material/analytics:", key="analyse")
        st.button("مسح", type="tertiary", icon=":material/close:", key="clear", on_click=clear_review,
                  disabled=not st.session_state["review"])
    with st.container(key="examples", gap="small"):
        st.caption("أو جرّب مثالًا:")
        # One row of chips; on a phone it scrolls sideways instead of taking several rows.
        with st.container(horizontal=True, wrap=False, gap="small", key="example-row"):
            for key, example in EXAMPLES.items():
                st.button(example.button, key=f"example-{key}", on_click=use_example, args=(key,))

    review = st.session_state["review"]
    if not review.strip():
        with st.container(border=True, key="result-empty"):
            st.caption(f"{icon('edit_note')} ستظهر النتيجة هنا: مشاعر المراجعة وجانبها الرئيسي، "
                       "ومدى ثقة النموذج في كلٍّ منهما.", unsafe_allow_html=True)
    else:
        try:
            with st.spinner("جارٍ التحليل…"):
                result = run_prediction(review, version)
        except InvalidReview as err:
            st.warning(err.message, icon=":material/edit_note:")
        except FileNotFoundError:
            st.error(MISSING_MODELS, icon=":material/error:")
        else:
            render_result(result)
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
        parts.append(f"يتفوق على النموذجين البسيطين ({ltr('Dummy')})" if all(dummies.values())
                     else f"لا يتفوق على كل النماذج البسيطة ({ltr('Dummy')})")
    return "، و".join(parts) + "." if parts else ""


def render_methodology(version_label: str, card: dict) -> None:
    """Methodology and frozen results of one model version; every number comes from ``card``."""
    tasks = card.get("tasks", {})
    sentiment, aspect = tasks.get("sentiment", {}), tasks.get("aspect", {})
    test_rows = card.get("test_run", {}).get("test_rows")

    st.header(f"{version_label}: نتائج مجمَّدة من اختبار محجوز", anchor=False)
    st.caption(
        "قيست هذه النتائج مرة واحدة على مراجعات محجوزة لم تُستخدم في التدريب ولا في أي اختيار، "
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

    st.subheader("البيانات والترخيص", anchor=False)
    data_lines = [f"مراجعات {ltr('Google Maps')} لمقاهي الرياض، بلهجة سعودية غالبًا ({ltr('Al-Dossari, 2025')})."]
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

    st.subheader("ماذا تعني ثقة النموذج؟", anchor=False)
    st.markdown(
        "مستوى الثقة وصف لنطاق أعلى درجة يعطيها النموذج لتنبؤه، **وليس احتمالًا معايرًا**. "
        "الجدولان يبيّنان ما لوحظ مرة واحدة على الاختبار المحجوز: حصة المراجعات في كل مستوى، "
        "ونسبة التنبؤات الصحيحة فيه. هذه تكرارات رُصدت على بيانات سابقة، **وليست ضمانًا** لصحة أي تنبؤ جديد."
    )
    level_cols = st.columns(2)
    for col, title, spec in ((level_cols[0], "المشاعر", sentiment), (level_cols[1], "الجانب الرئيسي", aspect)):
        levels = spec.get("certainty", {}).get("test_levels")
        if levels:
            with col:
                st.markdown(f"**{title}**")
                st.table(_rows_for_levels(levels), hide_index=True, alt=f"مستويات الثقة على الاختبار المحجوز · {title}")

    st.subheader("حدود معروفة", anchor=False)
    st.markdown(bullets((
        "جانب رئيسي واحد لكل مراجعة؛ لا يحلل النموذج مشاعر كل جانب مذكور في النص.",
        f"النطاق مقاهي الرياض على {ltr('Google Maps')}؛ لم يُقس الأداء على مطاعم أو لهجات أخرى.",
        "لا يكشف النموذج السخرية، والنفي لا يُلتقط إلا من تتابع الكلمات والحروف.",
        "النص الوصفي الخالي من الرأي قد يأخذ وسمًا إيجابيًا أو سلبيًا غير مقصود.",
        "نص من حروف عربية بلا معنى لا يُرفض دائمًا؛ ثقة النموذج المنخفضة هي الإشارة الوحيدة حينها.",
    )))

    with st.expander("تفاصيل تقنية", icon=":material/tune:"):
        lines = []
        for name, spec in (("المشاعر", sentiment), ("الجانب", aspect)):
            cfg = spec.get("configuration")
            if cfg:
                params = "، ".join(f"`{k} = {v}`" for k, v in cfg.get("params", {}).items())
                lines.append(f"**{name}:** التمثيل `{cfg['representation']}`، النموذج `{cfg['model']}`، "
                             f"أوزان الفئات `{cfg['class_weight']}`" + (f"، {params}" if params else ""))
        if card.get("fingerprint"):
            lines.append(f"بصمة البروتوكول المجمَّد: `{card['fingerprint'][:8]}`")
        versions = card.get("versions")
        if versions:
            lines.append("الإصدارات: " + "، ".join(f"`{k} {v}`" for k, v in versions.items()))
        lines.append("التقرير الكامل: `reports/final_test.md`")
        st.markdown(bullets(lines))


# --------------------------------------------------------------------------- page

st.set_page_config(page_title="تحليل مراجعات المقاهي", page_icon=":material/local_cafe:", layout="centered")
st.html(APP_DIR / "style.css")
init_state()

st.title("تحليل مراجعات المقاهي", anchor=False)
st.markdown("اكتب مراجعة عربية عن مقهى، ويقدّر النموذج **مشاعرها** و**جانبها الرئيسي**.")
st.caption("مشروع تعليمي، وليس نظامًا إنتاجيًا. النموذج مدرَّب على مراجعات مقاهي الرياض فقط.")

analyse_tab, method_tab = st.tabs(
    [TAB_ANALYSE, TAB_METHOD], default=TAB_METHOD if st.query_params.get("view") == "method" else None
)
with analyse_tab:
    render_analysis(DEFAULT_VERSION)
with method_tab:
    try:
        card = get_card(DEFAULT_VERSION)
    except FileNotFoundError:
        st.error("بطاقة النموذج غير موجودة: `models/model_card.json`.", icon=":material/error:")
    else:
        render_methodology(VERSIONS[DEFAULT_VERSION], card)

# Arabic page language for screen readers and hyphenation; the direction is set in style.css.
st.html("<script>document.documentElement.lang = 'ar';</script>", unsafe_allow_javascript=True)
