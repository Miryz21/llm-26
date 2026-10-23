# Ответы RAG (markdown-c400-o60-h-hybrid-bge-m3, top-5, без перевода)

| Метрика | Значение |
|---|---|
| ключевые факты в ответе | 0.75 |
| ответы с цитатами | 1.00 |
| цитаты из нужной статьи | 1.00 |
| среди цитат есть фрагмент-доказательство | 0.89 |
| отказ на вопросах вне корпуса | 2/2 |
| генерация, с (среднее) | 2.0 |
| токенов промпта (среднее) | 1503 |

## Q01. Из каких двух основных компонентов состоит Direct3D и за что отвечает каждый из них?

Direct3D состоит из двух основных компонентов:
1. **Direct 3D Variational Auto-Encoder (D3D-VAE)** — отвечает за эффективное кодирование высокоразрешенных 3D-форм в компактное и непрерывное пространство латентных трипланов, используя полунепрерывную стратегию выборки поверхности для сохранения деталей геометрии ([4]).
2. **Direct 3D Diffusion Transformer (D3D-DiT)** — моделирует распределение закодированных 3D-латентов и интегрирует позиционную информацию из трёх триплановых карт, обеспечивая генерацию 3D-контента ([4]).

**Источники:**

- [4] Direct3D, Abstract, стр. 1 — «Our approach comprises two primary components: a Direct 3D Variational Auto-Encoder ( **D3D-VAE** ) and a Direct 3D Diffusion Transformer ( **D3D-DiT** ).»

*Эталон:* D3D-VAE кодирует 3D-форму (облако точек) в компактное латентное триплановое пространство; D3D-DiT — диффузионный трансформер, генерирующий латенты в этом пространстве по входному изображению.

*Ключевые факты:* 0.67 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.98 с

## Q02. Что такое semi-continuous occupancy в D3D-VAE и зачем она нужна?

**Semi-continuous occupancy** в D3D-VAE — это подход, при котором occupancy (заполненность) точки в 3D пространстве определяется не только как бинарное значение (0 или 1), а как **континуальное значение от 0 до 1** для точек, близких к поверхности объекта (если расстояние до поверхности меньше порога *s* = 5121).

Этот метод используется, чтобы **сгладить резкие изменения градиентов** при обучении модели на точках, лежащих рядом с поверхностью объекта, где бинарная occupancy может вызывать проблемы с оптимизацией [2, 4]. В остальных случаях occupancy остаётся бинарной (0 или 1).

Также в фрагменте [3] отмечается, что данный подход улучшает качество реконструкции тонких структур, где использование только бинарной occupancy приводит к неудовлетворительным результатам.

**Источники:**

- [2] Direct3D, 3 Methods > 3.1 Direct 3D Variational Auto-Encoder, стр. 5, 6 — «Specifically, given a query point **x** in 3D space, when its distance to the surface is greater than a small threshold value _s_ = 5121, the occupancy value remains either 0 or 1.»
- [3] Direct3D, A Appendix > A.1 Ablation Studies, стр. 10 — «**Semi-continuous surface sampling strategy.** We adopt a semi-continuous surface sampling strategy during the training of D3D-VAE to alleviate the optimization difficulty caused by abrupt changes in occupancy near the object surface.»
- [4] Direct3D, 3 Methods > 3.1 Direct 3D Variational Auto-Encoder, стр. 4, 5 — «Meanwhile, we design a semi-continuous surface sampling strategy that utilizes both continuous and discrete supervision to ensure the high-frequency geometric details of the decoded 3D shape.»

*Эталон:* Вблизи поверхности (|sdf| < s, s = 1/512) occupancy принимает непрерывные значения от 0 до 1, дальше — бинарные 0/1; это сглаживает резкие изменения градиента у поверхности и облегчает оптимизацию.

*Ключевые факты:* 0.67 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.58 с

## Q03. Какие предобученные энкодеры изображений использует D3D-DiT для пиксельного и семантического выравнивания?

D3D-DiT использует **DINO-v2 (ViT-L/14)** для **пиксельного выравнивания** [3] и **CLIP (ViT-L/14)** для **семантического выравнивания** [4].

**Источники:**

- [3] Direct3D, 3 Methods > 3.2 Image-conditioned Direct 3D Diffusion Transformer, стр. 6, 7 — «We employ the pre-trained DINO-v2 [36] (ViT-L/14) as the pixel-level image encoder, which has been revealed in previous work [1] to outperform other pre-trained vision models in extracting structural information beneficial for 3D tasks.»
- [4] Direct3D, 3 Methods > 3.2 Image-conditioned Direct 3D Diffusion Transformer, стр. 7 — «We employ the pre-trained CLIP [40] visual model (ViT-L/14) to extract semantic image tokens **c** s from the conditional images, and then utilize a cross-attention layer within each DiT block to facilitate the interaction between **c** s…»

*Эталон:* Пиксельный уровень — DINO-v2 (ViT-L/14), токены конкатенируются с латентами в self-attention; семантический уровень — CLIP ViT-L/14 через cross-attention.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 0.98 с

## Q04. Какой сэмплер, сколько шагов и какой guidance scale используются при инференсе Direct3D?

При инференсе **Direct3D** используется **DDIM** с **50 шагами** и **guidance scale = 7.5** [1].

**Источники:**

- [1] Direct3D, 4 Experiments > 4.1 Implementation Details, стр. 8 — «During inference, we apply 50 steps of DDIM [48] with the guidance scale set to 7.5.»

*Эталон:* 50 шагов DDIM с guidance scale 7.5.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 0.75 с

## Q05. На основе какой видео-диффузионной модели построен ViewCrafter и как в неё подаются рендеры облака точек?

ViewCrafter построен на основе видео-диффузионной модели **DynamiCrafter** (open-sourced Image-to-Video diffusion model) [4].

Рендеры облака точек используются как условие для управления позой камеры: они преобразуются в **Plücker координаты** (описание движения на пиксельном уровне), которые затем конкатенируются с шумом в пространстве латентных переменных [5].

**Источники:**

- [4] ViewCrafter, 3 METHOD > 3.1 Preliminary: Video Diffusion Models, стр. 3, 4 — «In this work, we build our model based on an open-sourced Image-to-Video (I2V) diffusion model DynamiCrafter [18], which is capable of creating dynamic videos from a single input image.»
- [5] ViewCrafter, 4 EXPERIMENTS > 4.4 Ablation Study, стр. 10 — «Except for the pose condition strategy, we maintain the rest of the architecture of the Pl¨ucker model to be identical to ViewCrafter, and train the model at 320 × 512 resolution, following the training details reported in Section.»

*Эталон:* На image-to-video модели DynamiCrafter; рендеры облака точек кодируются VAE-энкодером и конкатенируются с шумом по каналам, референс дополнительно идёт через CLIP и cross-attention.

*Ключевые факты:* 0.33 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.39 с

## Q06. Как ViewCrafter реконструирует облако точек по одному или нескольким изображениям?

В предоставленных фрагментах нет прямого описания **процесса реконструкции облака точек** из одного или нескольких изображений в ViewCrafter.
Однако упоминается, что метод использует **точечные облака как явный приоритет** ([2]) и что **итеративный процесс** ([4]) включает обновление облака точек после синтеза новых видов:

- Начинается с **исходного точечного облака** (возможно, из входных данных или предобработки).
- После синтеза новых видов ([4]) они **обратнопроецируются** (*backproject*) на текущее облако точек, обновляя его.
- Процесс повторяется для получения более полного облака ([4]).

Точный алгоритм реконструкции не раскрыт.

**Источники:**

- [2] ViewCrafter, 5 CONCLUSION AND LIMITATION, стр. 12 — «This work presents ViewCrafter, a novel view synthesis framework that combines video diffusion models and point cloud priors for high-fidelity and accurate novel view synthesis.»
- [4] ViewCrafter, 3 METHOD > 3.4 Iterative View Synthesis and Camera Trajectory Planning, стр. 5 — «Subsequently, we backproject and align the generated novel view _I_ nbv onto the current point cloud _P_ curr, and designate _C_ nbv as the new _C_ curr, then repeat the aforementioned process until the predicted poses reach the predefined…»

*Эталон:* Плотной стерео-моделью DUSt3R: она строит point maps и карты уверенности, фокусное расстояние находится алгоритмом Вейсфельда; одно изображение дублируется в пару.

*Ключевые факты:* 0.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.14 с

## Q07. Как устроен алгоритм планирования траектории камеры в ViewCrafter и с какими параметрами он запускался в абляции?

Алгоритм планирования траектории камеры в **ViewCrafter** начинается с **инициализации** в одной из камерных позиций из входных изображений (*C_ref*). Затем для каждого шага алгоритм:

1. **Сэмплирует** *K* кандидатов камерных позиций (*C_can*) из поискового пространства (четверть-сферы с равномерно распределёнными позициями) вокруг текущей позиции (*C_curr*).
2. **Рендерит** маски кандидатов (*M_can*), где 1 — это области затенения/отсутствия, а 0 — заполненные области.
3. **Оптимизирует** следующую позицию камеры с помощью утилитарной функции *F(·)*, которая минимизирует площадь больших дырок (отклонение от порога Θ) и максимизирует видимость скрытых областей.

В абляции алгоритм запускался с параметрами:
- *N* = 3 (количество шагов или итераций, вероятно),
- *K* = 5 (число кандидатов камерных позиций на шаг),
- Θ = 0.6 (порог для допустимой площади дырок в маске).
Поисковое пространство — четверть-сфера с радиусом, равным глубине центрального пикселя входного изображения [1, 5].

**Источники:**

- [1] ViewCrafter, 3 METHOD > 3.4 Iterative View Synthesis and Camera Trajectory Planning, стр. 5 — «We then establish a utility function [64] _F_ ( · ) to determine the optimal camera pose for the subsequent step, defined as: where _C ∈ C_ can, _MC ∈ M_ can, and sum( _MC_ ) = � _Wu_ =0 � _Hv_ =0_MC(u, v_).Theutilityfunctionhelpsidentify…»
- [5] ViewCrafter, 4 EXPERIMENTS > 4.4 Ablation Study, стр. 11 — «We begin with exploring the left half area of the searching space, with the parameters for camera trajectory planning set to _N_ = 3, _K_ = 5, and Θ = 0.6.»

*Эталон:* Next-Best-View: из пространства поиска (четверть сферы) сэмплируются K кандидатных поз, по маскам дыр утилити-функция выбирает позу, открывающую достаточно, но не слишком много пропусков (порог Θ); затем генерация, обратная проекция и обновление облака. В абляции N = 3, K = 5, Θ = 0.6.

*Ключевые факты:* 0.67 · *цитаты из нужной статьи:* 1.00 · *генерация:* 3.61 с

## Q08. На каких данных и в каком объёме обучали ViewCrafter?

ViewCrafter обучали на смешанном датасете, состоящем из **DL3DV** и **RealEstate10K**.
В процессе обучения использовали **632 152 видеопары**, которые были получены из видеоклипов по 25 кадров каждый [2].

**Источники:**

- [2] ViewCrafter, 4 EXPERIMENTS > 4.1 Implementation Details, стр. 6 — «In the first stage, we train the ViewCrafter model at a resolution of 320 × 512, with the frame length set to 25.»

*Эталон:* На смеси DL3DV и RealEstate10K: клипы по 25 кадров, рендеры облаков точек через DUSt3R и PyTorch3D — всего 632 152 пары видео.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.1 с

## Q09. Что такое DreamEcom-400K и из каких шагов состоит пайплайн его построения?

DreamEcom-400K — это высококачественный датасет для задач генерации фонов в сценариях e-commerce, содержащий **400 000 разнообразных образцов**, включающих:
- изображения товаров,
- двуязычные текстовые промпты (английский/китайский),
- маски экземпляров продуктов,
- фоновые ссылочные изображения [1, 2].

**Пайплайн построения** включает следующие шаги:
1. Сборка данных (не указано в контексте, но подразумевается подготовка исходных изображений).
2. Создание масок экземпляров продуктов (не описано подробно, но маски используются в дальнейшем).
3. **Шаг 4**: Использование модели редактирования BAGLE для генерации фоновых ссылочных изображений — удаление продуктов и их теней/отражений с высокой точностью [2].
4. Формирование финальных образцов датасета с учетом всех компонентов [2].

В контексте не описаны детали шагов 1–3, но основной этап — это редактирование BAGLE для получения фонов [2].

**Источники:**

- [1] DreamPainter, Abstract, стр. 1 — «To address these challenges, we introduce DreamEcom-400K, a high-quality e-commerce dataset containing accurate product instance masks, background reference images, text prompts, and aesthetically pleasing product images.»
- [2] DreamPainter, 3 Dataset, стр. 3, 4 — «2 (left), each sample includes a commodity image, bilingual (Chinese-English) prompts, instance masks, and background reference images, establishing a robust foundation for background generation in e-commerce scenarios.»

*Эталон:* Датасет из 400k e-commerce примеров (изображение, двуязычный промпт, маска, референс фона). Шаги: LLM строит промпты, Seedream3 генерирует изображения, GroundingDINO + SAM дают маски, BAGEL удаляет товар и строит референс фона.

*Ключевые факты:* 0.33 · *цитаты из нужной статьи:* 1.00 · *генерация:* 3.12 с

## Q10. На какой базовой модели построен DreamPainter и почему выбрана именно она?

DreamPainter построен на базе **CogView4** [2].
Выбор обоснован подходящим размером модели и её поддержкой двух языков (китайского и английского), что важно для задач e-commerce.

**Источники:**

- [2] DreamPainter, 5 Experiments > 5.1 Experimental Setups, стр. 6 — «**Trainging Details.** DreamPainter is built upon CogView4 [35, 34], for its appropriate model size and dual-language (Chinese and English) support.»

*Эталон:* На CogView4 — из-за подходящего размера модели и поддержки двух языков (китайского и английского).

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 0.86 с

## Q11. Как устроено двухэтапное обучение DreamPainter?

Двухэтапное обучение в **DreamPainter** устроено так:

1. **Этап 1 (текстовый контроль)**:
   Обучение ведётся только по текстовым промптам. Оптимируются параметры *контрольного блока* (6 слоёв DIT Blocks), остальные параметры заморожены. Используется фиксированный learning rate 4e-5 [2, 4].

2. **Этап 2 (адаптация с учётом референса)**:
   Замораживаются параметры контрольного блока и оригинальной модели. Вводится LoRA (rank=256) в DiT блоки. Референсные изображения включаются с вероятностью 50% и подвергаются случайному изменению размера/обрезке. Применяется cosine annealing с пиковым learning rate 8e-5 [2, 4].

**Источники:**

- [2] DreamPainter, 5 Experiments > 5.1 Experimental Setups, стр. 6 — «A cosine annealing learning rate schedule was adopted, with a peak learning rate of 8e-5.»
- [4] DreamPainter, 4 Methodology > 4.2 Two-Stage Training Strategy, стр. 5 — «**Stage 2: Reference-aware adaptation.** In the second stage, we freeze the control branch and original model parameters, then introduce learnable LoRA into the DiT blocks.»

*Эталон:* Этап 1: обучается только control branch (6 DiT-блоков) с текстовым управлением, ~1k GPU-часов. Этап 2: control branch и база заморожены, добавляется LoRA ранга 256, референсы подаются в 50% примеров, ~700 GPU-часов.

*Ключевые факты:* 0.67 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.34 с

## Q12. Чем LoRA scaling отличается от attention modulation в DreamPainter при малых значениях масштаба?

При малых значениях масштаба (< 0.4) **LoRA scaling** приводит к резкому ухудшению качества сгенерированных изображений из-за несоответствия между обучением и инференсом: модель возвращается к состоянию после первого этапа обучения (T2I inpainting), не распознавая входные паттерны ссылочного изображения [1].

**Attention modulation**, напротив, при таких значениях продолжает увеличивать случайность без значительного ухудшения качества, так как модуляция внимания не нарушает общую структуру модели [1].

**Источники:**

- [1] DreamPainter, 5 Experiments > 5.4 Analyses, стр. 10 — «To assess those methods, we conducted the following experiments: (1) Fixing the LoRA scale at 1.0, we set the Attention scale to 0.8, 0.6, 0.4, 0.2, and 0.0 respectively; (2) Fixing the Attention scale at 1.0, we set the LoRA scale to 0.8,…»

*Эталон:* При масштабе ≥ 0.4 оба способа регулируют случайность; ниже 0.4 attention modulation продолжает работать без потери качества, а уменьшение LoRA scale резко ухудшает изображения: модель откатывается к состоянию после первого этапа и не понимает формат референса.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.68 с

## Q13. Из каких этапов состоит синтетическая аугментация данных в работе Preserving Product Fidelity?

Синтетическая аугментация данных в работе состоит из трёх этапов:
1. **Генерация новых ракурсов** с использованием image-to-video diffusion моделей [1].
2. **Разделение фона и объекта** (дизентангlement) путём маскирования и outpainting для замены фона [1].
3. **Создание негативных примеров** (включая контрфактуальные образцы) для улучшения обобщения модели [1, 4].

**Источники:**

- [1] Preserving Product Fidelity, 2 METHOD > 2.1 SYNTHETIC DATA AUGMENTATION, стр. 2, 3 — «We generate novel viewpoints using image-to-video diffusion models (Bar-Tal et al., 2024), augmenting the training data with diverse product perspectives (Appendix A.1 Table 2).»
- [4] Preserving Product Fidelity, A APPENDIX / SUPPLEMENTAL MATERIAL > A.1 SYNTHETIC AUGMENTATION TECHNIQUES, стр. 5, 8 — «Figure 2: Novel view generated using image-to-video diffusion.»

*Эталон:* Генерация новых ракурсов через image-to-video диффузию, смена фона маскированным outpainting (маска сегментации + промпты от LLM) и добавление негативов/контрфактуалов.

*Ключевые факты:* 0.67 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.45 с

## Q14. Какое соотношение позитивных и негативных примеров использовали в Preserving Product Fidelity и что такое контрфактуалы?

В статье используется соотношение **2:1 позитивных к негативным примерам** для улучшения качества и сохранения фidelity [2].

Контрфактуалы — это **изображения с тем же фоном, что и у оригинальных продуктов, но с другими объектами, нарисованными на месте продукта** (в маскированной области). Они помогают сохранить приоритеты классов, корректно рендерить фон/первый план и уменьшить артефакты [2]. Примеры контрфактуалов показаны на **Figure 4(b)**.

**Источники:**

- [2] Preserving Product Fidelity, 2 METHOD > 2.1 SYNTHETIC DATA AUGMENTATION > 2.1.2 NEW CONTEXT IMAGES, стр. 2, 3 — «We observed this gave the following benefits: (1) preserve class priors in the base generation model, (2) accurately render non-product objects in the background/foreground (3) render products with reduced artifacts in the image.»

*Эталон:* 2:1 (позитивы:негативы) — выше качество и точность ценой разнообразия. Контрфактуалы — изображения с тем же фоном, где в маску товара вписаны другие объекты.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.59 с

## Q15. Какие результаты человеческой оценки получил метод Preserving Product Fidelity по сравнению с DreamBooth + LoRA?

По результатам человеческой оценки метод **Preserving Product Fidelity** показал **17.40% проходов по изображению** и **45.5% проходов по продукту** на наборе мебели, в то время как **DreamBooth + LoRA** имел **10.00%** и **24.00%** соответственно ([3]).

Метод значительно превзошёл базовый подход по обоим показателям ([3]).

**Источники:**

- [3] Preserving Product Fidelity, 3 RESULTS > 3.2 QUANTITATIVE RESULTS, стр. 4 — «|Dataset|Overall Per Image pass rate↑|Per Product pass rate↑| |---|---|---| |Furniture (Ours)|**17.40%**|**45.5%**| |Furniture (DreamBooth)|10.00%|24.00%| Table 1: Human Evaluation results on multiple datasets created by mixing several…»

*Эталон:* Доля прошедших изображений 17.4% против 10%, доля товаров хотя бы с одним прошедшим изображением 45.5% против 24%; изображение проходит, если все 8 вопросов получили «да» по большинству из 3 оценщиков.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.41 с

## Q16. Как ранг LoRA (1 против 64) влияет на точность товара и разнообразие сцен в Preserving Product Fidelity?

Ранг LoRA **1** обеспечивает **высокую точность товара** быстрее (уже на **700 шагах обучения**), но при этом **теряет разнообразие сцен**: модели начинают генерировать простые, однообразные композиции и даже переобучаются, воспроизводя обучающие данные при дальнейшем обучении [2].

Ранг LoRA **64** обучается дольше (до **1800 шагов**), но лучше сохраняет **разнообразие и реалистичность сцен**, включая уникальные ракурсы и сложные окружения, хотя и он со временем теряет часть разнообразия при изучении деталей товара [1, 2].

**Источники:**

- [1] Preserving Product Fidelity, A APPENDIX / SUPPLEMENTAL MATERIAL > A.5 LORA RANK ABLATION QUALITATIVE RESULTS, стр. 11, 12 — «LoRA rank 1 achieves high product fidelity in much fewer training steps but generates less diverse scenes, before eventually overfitting and beginning to reproduce training data regardless of prompt.»
- [2] Preserving Product Fidelity, A APPENDIX / SUPPLEMENTAL MATERIAL > A.5 LORA RANK ABLATION QUALITATIVE RESULTS, стр. 11 — «The rank 1 model learned the appearance of the desk much quicker than rank 64 model (the samples display high product fidelity at 700 steps as opposed to 1800 steps 8).»

*Эталон:* Ранг 1 выучивает товар быстрее (700 шагов против 1800), но генерирует простые сцены и переобучается; ранг 64 учится дольше, зато сохраняет разнообразие и сложность сцен.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.08 с

## Q17. В каких из работ используется LoRA и для чего?

LoRA используется в следующих фрагментах:

1. **DreamPainter** ([1], [4]):
   - LoRA применяется для интеграции и модуляции влияния ссылочного (reference) изображения в процессе генерации.
   - В работе используется **LoRA scaling** для контроля степени влияния ссылочного изображения на результат: уменьшение масштаба LoRA снижает соответствие ссылочному образу, но может ухудшить качество изображения ([1]).
   - LoRA оптимизируется на этапе **второго стадии обучения** для работы с ссылочными изображениями ([4]).

2. **Preserving Product Fidelity** ([2], [3]):
   - LoRA используется для **fine-tuning** модели с целью сохранения точности воспроизведения объекта (продукта) и балансировки между детализацией и разнообразием сцен.
   - Исследуется зависимость от **LoRA rank** (ранга LoRA): низкий ранг (например, 1) быстрее достигает высокой точности, но приводит к переобучению и потере разнообразия; высокий ранг (например, 64) поддерживает разнообразие, но требует больше шагов обучения ([2], [3]).

**Источники:**

- [1] DreamPainter, 5 Experiments > 5.4 Analyses, стр. 10 — «To assess those methods, we conducted the following experiments: (1) Fixing the LoRA scale at 1.0, we set the Attention scale to 0.8, 0.6, 0.4, 0.2, and 0.0 respectively; (2) Fixing the Attention scale at 1.0, we set the LoRA scale to 0.8,…»
- [2] Preserving Product Fidelity, A APPENDIX / SUPPLEMENTAL MATERIAL > A.5 LORA RANK ABLATION QUALITATIVE RESULTS, стр. 11, 12 — «LoRA rank 1 achieves high product fidelity in much fewer training steps but generates less diverse scenes, before eventually overfitting and beginning to reproduce training data regardless of prompt.»
- [3] Preserving Product Fidelity, A APPENDIX / SUPPLEMENTAL MATERIAL > A.5 LORA RANK ABLATION QUALITATIVE RESULTS, стр. 11 — «The rank 1 model learned the appearance of the desk much quicker than rank 64 model (the samples display high product fidelity at 700 steps as opposed to 1800 steps 8).»
- [4] DreamPainter, 4 Methodology > 4.3 Flexible Control of Reference Images, стр. 6 — «**LoRA scaling.** Since the LoRA parameters are optimized specifically for reference integration in stage 2, we can scale their output to modulate reference influence: where _ϕ_ ( · ) denotes the linear layer in original DiT blocks, _s_ is…»

*Эталон:* DreamPainter — LoRA ранга 256 на втором этапе для учёта референсного изображения (её масштаб регулирует влияние референса); Preserving Product Fidelity — LoRA-дообучение в стиле DreamBooth под конкретный товар.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 3.24 с

## Q18. Какие работы используют image-to-video или видео-диффузионные модели и с какой целью?

В предоставленных фрагментах упоминаются следующие работы, использующие **image-to-video** или **видео-диффузионные модели**:

1. **ViewCrafter** использует **image-to-video diffusion model DynamiCrafter** для создания динамических видео из одного входного изображения [1]. Цель — синтез новых ракурсов (новых видов) на основе скудных или одиночных входных данных.

2. В работе по **Preserving Product Fidelity** применяется **image-to-video diffusion** в рамках пайплайна аугментации данных для создания синтетических тренировочных данных [2]. Это позволяет обойти ограничения реальных данных при задаче реконтекстуализации изображений продуктов.

3. В **ViewCrafter** видео-диффузионные модели интегрируются с реконструированными точечными облаками для **точечно-условного синтеза видео** с целью генерации высокофидельных и последовательных новых ракурсов [5]. Также они используются для точного контроля 6 степеней свободы камеры и прогрессивного расширения области покрытия новых видов.

**Источники:**

- [1] ViewCrafter, 3 METHOD > 3.1 Preliminary: Video Diffusion Models, стр. 3, 4 — «In this work, we build our model based on an open-sourced Image-to-Video (I2V) diffusion model DynamiCrafter [18], which is capable of creating dynamic videos from a single input image.»
- [2] Preserving Product Fidelity, ABSTRACT, стр. 1 — «We present a framework for high-fidelity product image recontextualization using text-to-image diffusion models and a novel data augmentation pipeline.»
- [5] ViewCrafter, 1 INTRODUCTION, стр. 1, 2 — «Leveraging the explicit 3D information from the point cloud and the generative capabilities of video diffusion models, our method enables precise control of 6 DoF camera poses and generates high-fidelity, consistent novel views.»

*Эталон:* ViewCrafter — видео-диффузия, обусловленная рендерами облака точек, для синтеза новых ракурсов сцены; Preserving Product Fidelity — image-to-video диффузия для генерации новых ракурсов товара в обучающих данных.

*Ключевые факты:* 0.50 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.82 с

## Вне корпуса: Какой FID получает Stable Diffusion 3 на MS-COCO?

В предоставленных фрагментах нет ответа на этот вопрос.

*Отказ:* да

## Вне корпуса: Сколько параметров у модели Sora и на каких данных её обучали?

В предоставленных фрагментах нет информации о модели **Sora** или её параметрах, а также о данных, на которых её обучали.

*Отказ:* да

