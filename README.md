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
python -m pytest -q
```

مخرجات التحقق المنشورة: [`reports/data_quality.md`](reports/data_quality.md) (تقرير الجودة وأثر كل خطوة على عدد السجلات) و[`reports/lineage.json`](reports/lineage.json) (سجل كل استبعاد وكل حل لوسوم المكررات). إعادة التشغيل تنتج الملفات نفسها بايتًا ببايت.

## الترخيص

الكود متاح بترخيص MIT (انظر [`LICENSE`](LICENSE)). البيانات تخضع لترخيص مصدرها.
