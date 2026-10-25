"""Conservative repair rules, exercised with variants observed in the source file."""

import pytest

from arabic_reviews.validate import clean_whitespace, has_arabic, strip_label_suffix, strip_translation


@pytest.mark.parametrize(
    "raw, text, aspect, sentiment",
    [
        ("احببته\n\n\nLoved it;الجلسات;ايجابي", "احببته\n\n\nLoved it", "الجلسات", "ايجابي"),
        ("الطلب تأخر كثير;الخدمة;سلبى", "الطلب تأخر كثير", "الخدمة", "سلبي"),
        ("الخدمة بطيئة:الخدمة", "الخدمة بطيئة", "الخدمة", None),
        ("قهوتهم جدا لذيذة وطعمها حلو :المشروبات;ايجابي", "قهوتهم جدا لذيذة وطعمها حلو", "المشروبات", "ايجابي"),
        ("كل شي حلو:الخدمة;سلبى", "كل شي حلو", "الخدمة", "سلبي"),
        ("الاسعار معقولة;الاسعار;ايجابي\n.", "الاسعار معقولة", "الاسعار", "ايجابي"),
        ("متوسط\n\n …;المأكولات;محايد", "متوسط\n\n …", "المأكولات", "محايد"),
    ],
)
def test_end_anchored_label_suffix_is_removed_and_reported(raw, text, aspect, sentiment):
    assert strip_label_suffix(raw) == (text, aspect, sentiment)


@pytest.mark.parametrize(
    "natural",
    [
        "الخدمة ممتازة والاسعار",
        "افضل شي فيهم الخدمة",
        "رأيي باختصار: الخدمة",
        "المأكولات;المشروبات كلها ممتازة",
    ],
)
def test_aspect_words_in_natural_text_are_untouched(natural):
    assert strip_label_suffix(natural) == (natural, None, None)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("(تمت الترجمة بواسطة Google.) طعم مذهل  ببساطة واو ..", "طعم مذهل  ببساطة واو .."),
        ("(تمت الترجمة بواسطة Google الكعك ليست طازجة دائمًا", "الكعك ليست طازجة دائمًا"),
        (" احببته، مكان يستحق الزيارة\n\n\nLoved it", " احببته، مكان يستحق الزيارة"),
        ("الخدمة المقدمة لا تتجاوز بخير\n\n\nالخدمة provided don't exceed okay", "الخدمة المقدمة لا تتجاوز بخير"),
        (" مبالغ فيها.\n\n\nOverالاسعارd.", " مبالغ فيها."),
        (" frạbtshynw\n\n\nفرابتشينو لذيذ", "فرابتشينو لذيذ"),
        ("Bad branch of Starbucks in the world.\nحقيقة اسؤا فرع", "حقيقة اسؤا فرع"),
        ("اسواء فرع .. الأكل سيء ناشف 😑\nThe worst Starbucks branch. The food was not fresh", "اسواء فرع .. الأكل سيء ناشف 😑"),
        ("متوسط . كان يعمل\n\n …", "متوسط . كان يعمل\n\n …"),
        ("ممتاز لكن يحتاج إلى drive thru خدمة سيارات", "ممتاز لكن يحتاج إلى drive thru خدمة سيارات"),
        ("Loved it", "Loved it"),
    ],
)
def test_translation_segments_are_removed_only_when_arabic_text_remains(raw, expected):
    assert strip_translation(raw) == expected


def test_whitespace_and_invisible_characters_are_standardised():
    raw = " متوسط ​​. كان\r\n\r\n\tيعمل   صديقي‏ "
    assert clean_whitespace(raw) == "متوسط . كان\nيعمل صديقي"


def test_has_arabic_requires_an_arabic_letter():
    assert has_arabic("غالي") and not has_arabic("Loved it 😍 123")
