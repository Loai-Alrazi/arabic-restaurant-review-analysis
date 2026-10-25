"""One-page Arabic demo: type a cafe review, see the predicted main aspect and sentiment.

Run:  streamlit run app/streamlit_app.py
The page only calls ``arabic_reviews.predict``; it applies no text transformation of its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from arabic_reviews.predict import InvalidReview, MAX_CHARS, model_card, predict  # noqa: E402

EXAMPLES = {
    "قهوة ممتازة": "القهوة ممتازة والطعم رائع",
    "خدمة بطيئة": "الخدمة بطيئة والموظف ما رد علينا",
    "أسعار مرتفعة": "الأسعار غالية مقارنة بالكمية",
    "جلسة هادئة": "الجلسة هادئة والمكان مريح",
    "نفي": "ما عجبني الكيك",
}
NOTES = {
    "long_text": "المراجعة أطول من معظم مراجعات التدريب؛ قد تقل الدقة.",
    "very_short": "المراجعة قصيرة جدًا؛ النتيجة أقل ثباتًا.",
}
LEVEL_TEXT = {
    "high": "في هذا المستوى كانت معظم التنبؤات صحيحة في اختبار المشروع.",
    "medium": "في هذا المستوى أخطأ النموذج أحيانًا في اختبار المشروع.",
    "low": "النموذج غير محسوم هنا؛ يُنصح بقراءة المراجعة بنفسك.",
}

st.set_page_config(page_title="تحليل مراجعات المقاهي", page_icon="☕", layout="centered")
st.markdown(
    "<style>html, body, [data-testid='stAppViewContainer'] {direction: rtl; text-align: right;}"
    "textarea {direction: rtl; text-align: right;}</style>",
    unsafe_allow_html=True,
)

st.title("تحليل مراجعات المقاهي")
st.caption("يتنبأ بالجانب الرئيسي في المراجعة ومشاعرها. مشروع تعليمي، وليس نظامًا إنتاجيًا.")

if "review" not in st.session_state:
    st.session_state["review"] = ""
cols = st.columns(len(EXAMPLES))
for col, (name, text) in zip(cols, EXAMPLES.items()):
    if col.button(name):
        st.session_state["review"] = text

review = st.text_area("اكتب مراجعة عن مقهى", key="review", max_chars=MAX_CHARS, height=120)

if review.strip():
    try:
        result = predict(review)
    except InvalidReview as err:
        st.warning(err.message)
    except FileNotFoundError as err:
        st.error(str(err))
    else:
        left, right = st.columns(2)
        for col, title, task in ((left, "الجانب الرئيسي", result.aspect), (right, "المشاعر", result.sentiment)):
            with col:
                st.subheader(title)
                st.markdown(f"### {task.label}")
                st.write(f"مستوى اليقين: **{task.level_word}**")
                st.caption(LEVEL_TEXT[task.level])
                if task.second_label:
                    st.write(f"البديل الأقرب: {task.second_label}")
        for note in result.notes:
            st.info(NOTES[note])

card = model_card()
with st.expander("حدود الادعاء ونتائج الاختبار"):
    st.markdown(
        "- كل مراجعة لها **جانب رئيسي واحد** ومشاعر واحدة؛ لا يحلل النموذج كل الجوانب المذكورة في النص.\n"
        "- مستوى اليقين وصف لمدى الدرجة وليس احتمالًا معايرًا.\n"
        "- التدريب على مراجعات مقاهي الرياض من مصدر واحد؛ لهجات ومجالات أخرى لم تُختبر.\n"
        f"- نتيجة الاختبار المحجوز (Macro-F1): الجانب {card['tasks']['aspect']['test']['macro_f1']:.3f}، "
        f"المشاعر {card['tasks']['sentiment']['test']['macro_f1']:.3f}؛ "
        f"التطابق المشترك {card['joint_test_accuracy']:.3f}.\n"
        "- فئة «محايد» هي الأضعف (F1 ≈ "
        f"{card['tasks']['sentiment']['test']['per_class_f1']['محايد']:.2f})."
    )
