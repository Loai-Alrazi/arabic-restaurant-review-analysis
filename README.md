# تحليل مراجعات المقاهي العربية — تصنيف المشاعر والجانب الرئيسي

> **حالة المشروع:** قيد التنفيذ وفق [`PROJECT_PLAN.md`](PROJECT_PLAN.md). يُستكمل هذا الملف بالنتائج الفعلية بعد انتهاء التقييم.

مشروع تعلم آلة تطبيقي فردي يصنّف نص مراجعة عربية لمقهى إلى:

- **الجانب الرئيسي:** المأكولات، المشروبات، الخدمة، الجلسات، الاسعار.
- **المشاعر المرتبطة به:** ايجابي، سلبي، محايد.

المنهجية: صياغة المشكلة ← هندسة البيانات والتحقق من جودتها ← التقسيم أولًا ومنع التسريب ← خصائص وPipelines قابلة لإعادة الإنتاج ← خطوط أساس ثم تحسين مبرر ← اختبار نهائي لمرة واحدة ← واجهة عربية محلية.

## البيانات

- **المصدر:** Hmood Al-Dossari, *Multi-Aspect Sentiment Analysis of Arabic Café Reviews Using Machine and Deep Learning Approaches* — Zenodo, 2025. DOI: [10.5281/zenodo.17781916](https://doi.org/10.5281/zenodo.17781916).
- **الترخيص:** CC-BY-4.0. لا تُرفع البيانات إلى هذا المستودع؛ تُجلب من المصدر بسكربت يتحقق من البصمة.

- **صياغة المشكلة ومعنى الوسوم:** [`docs/PROBLEM_FRAMING.md`](docs/PROBLEM_FRAMING.md).

## التشغيل

```bash
python -m venv .venv            # Python 3.12
.venv/Scripts/python -m pip install -r requirements.txt   # على Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pip install -e . --no-deps

python -m arabic_reviews ingest     # تنزيل الملف من Zenodo عند غيابه والتحقق من البصمة → data/source_manifest.json
python -m arabic_reviews validate   # فحص المخطط والجودة والإصلاح المحافظ → data/processed/validated.csv (محلي)
python -m arabic_reviews split      # تجميع شبه المكررات والتقسيم → data/splits/split_v1.csv + reports/split.md
python -m arabic_reviews eda        # تحليل استكشافي على بيانات التطوير فقط → reports/eda.md + reports/figures/
python -m arabic_reviews experiments  # خطوط الأساس وتجارب الإزالة/الإضافة على 3×5 CV → reports/experiments.csv + reports/ablations.md
python -m arabic_reviews models     # عائلات النماذج والضبط المحدود وتحليل الأخطاء → reports/models.md + reports/error_analysis.md
python -m arabic_reviews protocol   # تجميد البروتوكول قبل الاختبار (يتطلب شجرة Git نظيفة) → reports/final_protocol.md + reports/protocol.json
python -m pytest -q
```

بروتوكول التقييم المجمّد قبل قراءة الاختبار المحجوز: [`reports/final_protocol.md`](reports/final_protocol.md) (الإعدادات النهائية، مستويات اليقين، المدى المتوقع ببذور CV جديدة، وإجراء الاختبار لمرة واحدة).

مخرجات التحقق المنشورة: [`reports/data_quality.md`](reports/data_quality.md) (تقرير الجودة وأثر كل خطوة على عدد السجلات) و[`reports/lineage.json`](reports/lineage.json) (سجل كل استبعاد وكل حل لوسوم المكررات). إعادة التشغيل تنتج الملفات نفسها بايتًا ببايت.

## التقسيم ومنع التسريب

- **التقسيم أولًا:** قبل أي تحليل موجّه أو نمذجة، تُجمَّع المراجعات شبه المكررة في `group_id` واحد دون استخدام الوسوم: تشابه Cosine ≥ 0.80 على TF-IDF أحرف (3–5)، أو قواعد نسخ تلتقط ما يفوته التشابه المتماثل: التطابق بعد حذف المسافات والسوابق، ثم — لنصوص من 4 كلمات فأكثر — تعديل حرفي طفيف، أو تغطية مقاطع الأحرف أو تسلسلات الكلمات للنص الأقصر داخل الأطول (نسخ مقتطعة أو معدّلة). العتبات مختارة بقراءة أزواج من كل نطاق تشابه، والتفاصيل في [`reports/split.md`](reports/split.md).
- **اختبار محجوز:** `StratifiedGroupKFold` على الوسم المشترك `aspect|sentiment` مع المجموعات؛ الطية الأولى (554 مراجعة، 20%) اختبار نهائي لا يُقرأ قبل التقييم النهائي لمرة واحدة، والباقي (2215) للتطوير مع 3 تكرارات من CV خماسي مجمّع وطبقي (بذور 42، 43، 44). أكبر فرق في نسب الفئات بين التطوير والاختبار 0.26 نقطة مئوية.
- **ضمانات آلية:** اختبارات تتحقق من عدم عبور أي مجموعة أو نص مطبَّع أو زوج شبيه بين التطوير والاختبار أو بين الطيات، وثبات التقسيم عند إعادة التشغيل (يُعاد توليده في CI ويُقارن بايتًا ببايت)، وأن كود التجارب يقرأ التطوير فقط عبر `split.load_dev()`.
- **تحليل استكشافي على التطوير فقط:** [`reports/eda.md`](reports/eda.md) — توزيع الوسوم، الأطوال، النفي والرموز التعبيرية، أكثر n-grams تمييزًا، وتقدير نسبة المراجعات متعددة الجوانب (≈ 10%).

## الترخيص

الكود متاح بترخيص MIT (انظر [`LICENSE`](LICENSE)). البيانات تخضع لترخيص مصدرها.
