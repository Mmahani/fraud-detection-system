# Portfolio Copy — LinkedIn and CV

Ready-to-edit descriptions based only on verified repository results. Add a date
or location if you want. Do not add claims that are not supported by the
repository.

## English

### LinkedIn headline

> Machine Learning Engineer | Fraud Detection & Explainable AI | Python · scikit-learn · SHAP · FastAPI

### LinkedIn project entry (short)

> **Explainable Fraud Detection — IEEE-CIS (Research Project)**
>
> Built an end-to-end fraud detection pipeline on 590k IEEE-CIS transactions:
> leakage-safe preprocessing, model comparison, a tuned Random Forest, OOF
> threshold selection, SHAP explanations, and a local FastAPI service. Achieved
> **0.7305 PR-AUC** on a retrospective test split and **0.7049 ± 0.0090** in
> 5-fold training-only cross-validation. Packaged the model with a SHA-256
> artifact manifest, 21 tests, and CI.

### LinkedIn project entry (longer)

> **Explainable Fraud Detection — IEEE-CIS (Research Project)**
>
> Fraud is a rare-event ranking problem: only 3.5% of transactions in this dataset
> are fraudulent, so accuracy is misleading. I built an end-to-end research
> pipeline covering data integration, leakage-safe preprocessing, baseline and
> model comparison (SGD, Random Forest, Gradient Boosting, XGBoost), tuning, and
> threshold selection from out-of-fold probabilities.
>
> The frozen Random Forest reaches **0.7305 PR-AUC** on a retrospective test split
> and **0.7049 ± 0.0090** in 5-fold training-only cross-validation. I added
> SHAP-based global and local explanations with class-1 alignment and additivity
> checks, then shipped a tested FastAPI service for `/predict` and `/explain`.
> The repository documents its limitations, data rights, and reproducibility
> controls instead of overstating readiness.

### CV bullets

- Built an end-to-end fraud detection pipeline on 590k IEEE-CIS transactions,
  covering integration, leakage-safe preprocessing, model comparison, tuning, and
  evaluation (**0.7305 test PR-AUC**; **0.7049 ± 0.0090** training-only 5-fold
  PR-AUC).
- Selected a **0.35** operating threshold from out-of-fold probabilities using an
  explicit, documented cost assumption; froze the model and packaged artifacts
  with a SHA-256 manifest and 21-test suite.
- Added SHAP-based global and local explanations with class-1 alignment and
  additivity checks; built a FastAPI inference service and GitHub Actions CI.

### One-line summary

> End-to-end fraud detection research project on IEEE-CIS: leakage-safe pipeline,
> tuned Random Forest (0.7305 test PR-AUC), SHAP explanations, and a local
> FastAPI service.

### Keyword line

> Fraud Detection · Imbalanced Classification · PR-AUC · scikit-learn · SHAP ·
> FastAPI · Model Evaluation · Reproducibility · Python

## العربية

### عنوان LinkedIn

> مهندس تعلّم آلي | كشف الاحتيال والذكاء الاصطناعي القابل للتفسير | Python · scikit-learn · SHAP · FastAPI

### وصف المشروع (مختصر)

> **كشف الاحتيال القابل للتفسير — IEEE-CIS (مشروع بحثي)**
>
> بنيت خط أنابيب متكاملًا لكشف الاحتيال على 590 ألف معاملة من IEEE-CIS: معالجة
> مسبقة آمنة من التسريب، ومقارنة نماذج، وغابة عشوائية مضبوطة، واختيار عتبة من
> احتمالات out-of-fold، وتفسيرات SHAP، وخدمة FastAPI محلية. حقق النموذج
> **0.7305 PR-AUC** على تقسيم اختبار استرجاعي، و**0.7049 ± 0.0090** في تحقق
> متقاطع خمسي على بيانات التدريب فقط. الحزمة تشمل manifest ببصمات SHA-256،
> و21 اختبارًا، وتكاملًا مستمرًا.

### نقاط السيرة الذاتية

- بناء خط أنابيب متكامل لكشف الاحتيال على 590 ألف معاملة IEEE-CIS، يشمل الدمج،
  والمعالجة الآمنة من التسريب، ومقارنة النماذج، والضبط، والتقييم
  (**0.7305 PR-AUC** على الاختبار؛ **0.7049 ± 0.0090** في تحقق خمسي على التدريب).
- اختيار عتبة تشغيل **0.35** من احتمالات out-of-fold وفق افتراض تكلفة موثّق، مع
  تجميد النموذج وتوثيق القطع الأثرية عبر SHA-256 و21 اختبارًا.
- إضافة تفسيرات SHAP عامة ومحلية مع التحقق من محاذاة الفئة الأولى والجمعية،
  وبناء خدمة FastAPI وتكامل مستمر عبر GitHub Actions.

### سطر واحد

> مشروع بحثي متكامل لكشف الاحتيال على IEEE-CIS: خط أنابيب آمن من التسريب، وغابة
> عشوائية مضبوطة (0.7305 PR-AUC)، وتفسيرات SHAP، وخدمة FastAPI محلية.

## Claims to avoid

Do **not** describe this project as:

- deployed to production or handling live traffic;
- saving money, reducing losses, or improving business KPIs;
- state-of-the-art or competition-winning;
- having calibrated probabilities;
- validated with a strict forward-in-time holdout;
- suitable for real customer decisions without further work.

The repository explicitly documents these limitations; the portfolio copy should
match it.
