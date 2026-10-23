# Ответы RAG (markdown-c400-o60-h-hybrid-bge-m3, top-5, без перевода)

| Метрика | Значение |
|---|---|
| ключевые факты в ответе | 0.74 |
| ответы с цитатами | 1.00 |
| цитаты из нужной статьи | 1.00 |
| среди цитат есть фрагмент-доказательство | 0.78 |
| отказ на вопросах вне корпуса | 2/2 |
| генерация, с (среднее) | 1.9 |
| токенов промпта (среднее) | 1478 |

## Q01. Из каких двух основных компонентов состоит Direct3D и за что отвечает каждый из них?

Direct3D состоит из двух основных компонентов:
1. **Direct 3D Variational Auto-Encoder (D3D-VAE)** — отвечает за эффективное кодирование высокоразрешенных 3D-форм в компактное и непрерывное пространство латентных трипланов, а также за декодирование геометрии с использованием полунепрерывной стратегии выборки поверхности ([5]).
2. **Direct 3D Diffusion Transformer (D3D-DiT)** — моделирует распределение закодированных 3D-латентных представлений и интегрирует позиционную информацию из трёх триплановых карт, обеспечивая генерацию 3D-форм ([5]).

**Источники:**

- [5] Direct3D, Abstract, стр. 1 — «Our approach comprises two primary components: a Direct 3D Variational Auto-Encoder ( **D3D-VAE** ) and a Direct 3D Diffusion Transformer ( **D3D-DiT** ).»

*Эталон:* D3D-VAE кодирует 3D-форму (облако точек) в компактное латентное триплановое пространство; D3D-DiT — диффузионный трансформер, генерирующий латенты в этом пространстве по входному изображению.

*Ключевые факты:* 0.67 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.97 с

## Q02. Что такое semi-continuous occupancy в D3D-VAE и зачем она нужна?

**Semi-continuous occupancy** в D3D-VAE — это подход, при котором occupancy (заполненность) точки в 3D пространстве определяется не только как бинарное значение (0 или 1), а также как **континуальное значение от 0 до 1**, когда точка находится близко к поверхности объекта (если её расстояние до поверхности меньше порога *s* = 5121).

Этот метод вводится для **сглаживания градиентов** при обучении модели, так как резкие изменения occupancy у поверхности могут мешать оптимизации. Таким образом, semi-continuous occupancy позволяет **обеспечить плавное обучение** и улучшить качество восстановления тонких структур [1, 2, 3, 4].

**Источники:**

- [1] Direct3D, 3 Methods > 3.1 Direct 3D Variational Auto-Encoder, стр. 5, 6 — «Specifically, given a query point **x** in 3D space, when its distance to the surface is greater than a small threshold value _s_ = 5121, the occupancy value remains either 0 or 1.»
- [2] Direct3D, A Appendix > A.1 Ablation Studies, стр. 10 — «**Semi-continuous surface sampling strategy.** We adopt a semi-continuous surface sampling strategy during the training of D3D-VAE to alleviate the optimization difficulty caused by abrupt changes in occupancy near the object surface.»
- [3] Direct3D, 3 Methods > 3.1 Direct 3D Variational Auto-Encoder, стр. 4, 5 — «Meanwhile, we design a semi-continuous surface sampling strategy that utilizes both continuous and discrete supervision to ensure the high-frequency geometric details of the decoded 3D shape.»
- [4] Direct3D, 1 Introduction, стр. 2, 3 — «Furthermore, to ensure precise supervision of the 3D occupancy grid, we adopt a semi-continuous surface sampling strategy, enabling the sampling and supervision of surface points in both continuous and discrete manners.»

*Эталон:* Вблизи поверхности (|sdf| < s, s = 1/512) occupancy принимает непрерывные значения от 0 до 1, дальше — бинарные 0/1; это сглаживает резкие изменения градиента у поверхности и облегчает оптимизацию.

*Ключевые факты:* 0.67 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.17 с

## Q03. Какие предобученные энкодеры изображений использует D3D-DiT для пиксельного и семантического выравнивания?

Для **пиксельного выравнивания** D3D-DiT использует предобученный энкодер **DINO-v2 (ViT-L/14)** [5].
Для **семантического выравнивания** — предобученный энкодер **CLIP (ViT-L/14)** [3].

**Источники:**

- [3] Direct3D, 3 Methods > 3.2 Image-conditioned Direct 3D Diffusion Transformer, стр. 7 — «We employ the pre-trained CLIP [40] visual model (ViT-L/14) to extract semantic image tokens **c** s from the conditional images, and then utilize a cross-attention layer within each DiT block to facilitate the interaction between **c** s…»
- [5] Direct3D, 3 Methods > 3.2 Image-conditioned Direct 3D Diffusion Transformer, стр. 6, 7 — «We employ the pre-trained DINO-v2 [36] (ViT-L/14) as the pixel-level image encoder, which has been revealed in previous work [1] to outperform other pre-trained vision models in extracting structural information beneficial for 3D tasks.»

*Эталон:* Пиксельный уровень — DINO-v2 (ViT-L/14), токены конкатенируются с латентами в self-attention; семантический уровень — CLIP ViT-L/14 через cross-attention.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.16 с

## Q04. Какой сэмплер, сколько шагов и какой guidance scale используются при инференсе Direct3D?

При инференсе **Direct3D** используется **DDIM** с **50 шагами** и **guidance scale = 7.5** [1].

**Источники:**

- [1] Direct3D, 4 Experiments > 4.1 Implementation Details, стр. 8 — «During inference, we apply 50 steps of DDIM [48] with the guidance scale set to 7.5.»

*Эталон:* 50 шагов DDIM с guidance scale 7.5.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 0.74 с

## Q05. На основе какой видео-диффузионной модели построен ViewCrafter и как в неё подаются рендеры облака точек?

ViewCrafter построен на основе **видео-диффузионной модели**, но конкретное название модели не указано в контексте. Однако в [1] и [4] подчёркивается, что это **точечно-условленная (point-conditioned) видео-диффузионная модель**, которая принимает на вход **рендеры облака точек** (point cloud render results) в качестве условий для генерации новых видов.

Рендеры облака точек используются как входные данные для модели, чтобы обеспечить точный контроль над позой камеры и генерацию высококачественных новых видов ([1], [4]).

**Источники:**

- [1] ViewCrafter, 4 EXPERIMENTS > 4.4 Ablation Study, стр. 10, 11 — «**Robustness for point cloud condition.** ViewCrafter utilizes point cloud render results as conditions, enabling highly accurate pose control.»
- [4] ViewCrafter, 1 INTRODUCTION, стр. 1, 2 — «Next, we render the point cloud according to the predicted pose sequence and synthesize novel views by ViewCrafter with the conditions of the rendered point cloud.»

*Эталон:* На image-to-video модели DynamiCrafter; рендеры облака точек кодируются VAE-энкодером и конкатенируются с шумом по каналам, референс дополнительно идёт через CLIP и cross-attention.

*Ключевые факты:* 0.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.81 с

## Q06. Как ViewCrafter реконструирует облако точек по одному или нескольким изображениям?

ViewCrafter реконструирует облако точек из одного или нескольких входных изображений, используя **точечную облако-условленную модель** на основе видео-диффузии. Вначале из входных изображений восстанавливается начальная **3D-геометрия** в виде точечного облака (реконструкция на основе входных данных). Затем это облако точек используется в качестве условия для генерации новых ракурсов.

Подробнее: в [4] описано, что метод начинает с **точечного облака**, полученного из входных изображений, и затем итеративно улучшает его, используя синтезированные ракурсы для расширения области покрытия. В [1] также упоминается, что ViewCrafter работает с **рендерингом точечного облака** как условием для генерации новых видов.

**Источники:**

- [1] ViewCrafter, 4 EXPERIMENTS > 4.4 Ablation Study, стр. 10, 11 — «**Text-to-3D generation results.** The leftmost column displays the input text prompt and the corresponding image generated by T2I model, while the subsequent columns show the consistent novel views produced by our ViewCrafter based on the…»
- [4] ViewCrafter, 1 INTRODUCTION, стр. 1, 2 — «Our method, **ViewCrafter** , accomplishes novel view synthesis by a _point-conditioned video diffusion model_ that generates high-fidelity and consistent videos under a novel view trajectory, conditioned on corresponding frames rendered…»

*Эталон:* Плотной стерео-моделью DUSt3R: она строит point maps и карты уверенности, фокусное расстояние находится алгоритмом Вейсфельда; одно изображение дублируется в пару.

*Ключевые факты:* 0.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.29 с

## Q07. Как устроен алгоритм планирования траектории камеры в ViewCrafter и с какими параметрами он запускался в абляции?

Алгоритм планирования траектории камеры в **ViewCrafter** использует **итеративный подход** с утилитарной функцией, которая оценивает подходящие позы камеры, минимизируя большие дыры и отклонения от порогового значения Θ, чтобы улучшить генерацию новых видов [1].

В абляции модель обучалась на разрешении **320 × 512** с использованием **Plücker координат** как альтернативного условия для сравнения, но основной метод применялся с **точечными облаками** как условием [3].

**Источники:**

- [1] ViewCrafter, 3 METHOD > 3.4 Iterative View Synthesis and Camera Trajectory Planning, стр. 5 — «where _C ∈ C_ can, _MC ∈ M_ can, and sum( _MC_ ) = � _Wu_ =0 � _Hv_ =0_MC(u, v_).Theutilityfunctionhelpsidentify a suitable camera pose that reveals an adequate area of occlusion and missing regions while avoiding poses that reveal…»
- [3] ViewCrafter, 4 EXPERIMENTS > 4.4 Ablation Study, стр. 10 — «Except for the pose condition strategy, we maintain the rest of the architecture of the Pl¨ucker model to be identical to ViewCrafter, and train the model at 320 × 512 resolution, following the training details reported in Section.»

*Эталон:* Next-Best-View: из пространства поиска (четверть сферы) сэмплируются K кандидатных поз, по маскам дыр утилити-функция выбирает позу, открывающую достаточно, но не слишком много пропусков (порог Θ); затем генерация, обратная проекция и обновление облака. В абляции N = 3, K = 5, Θ = 0.6.

*Ключевые факты:* 0.33 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.77 с

## Q08. На каких данных и в каком объёме обучали ViewCrafter?

ViewCrafter обучали на смешанном датасете, состоящем из **DL3DV** и **RealEstate10K**.
В процессе обучения было подготовлено **632 152 пар видео** (видео + соответствующие им рендеры точечных облаков) [2].

**Источники:**

- [2] ViewCrafter, 4 EXPERIMENTS > 4.1 Implementation Details, стр. 6 — «Our model was trained on a mixed dataset consisting of DL3DV [70] and RealEstate10K [7].»

*Эталон:* На смеси DL3DV и RealEstate10K: клипы по 25 кадров, рендеры облаков точек через DUSt3R и PyTorch3D — всего 632 152 пары видео.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.1 с

## Q09. Что такое DreamEcom-400K и из каких шагов состоит пайплайн его построения?

DreamEcom-400K — это высококачественный датасет из **400 000 разнообразных образцов** для генерации фонов в сценариях e-commerce, включающий:
коммерческие изображения, двуязычные (китайско-английские) промпты, инстанс-маски и фоновые референс-изображения [2, 3, 4].

Пайплайн его построения включает **4 этапа**:
1. Сбор данных (не упомянут в контексте, но подразумевается подготовка исходных изображений).
2. **Создание референс-фонов** с помощью модели BAGLE для семантической правки (например, удаление продукта и его теней/отражений) [2].
3. Генерация инстанс-масок и текстовых промптов для каждого образца.
4. Объединение всех компонентов (изображения, маски, промпты, референсы) в финальный датасет [2, 3].

**Источники:**

- [2] DreamPainter, 3 Dataset, стр. 3, 4 — «2 (left), each sample includes a commodity image, bilingual (Chinese-English) prompts, instance masks, and background reference images, establishing a robust foundation for background generation in e-commerce scenarios.»
- [3] DreamPainter, Abstract, стр. 1 — «To address these challenges, we introduce DreamEcom-400K, a high-quality e-commerce dataset containing accurate product instance masks, background reference images, text prompts, and aesthetically pleasing product images.»
- [4] DreamPainter, 6 Conclusion, стр. 10, 11 — «In this paper, we tackle the challenge of the background inpainting for e-commerce scenarios by introducing DreamPainter, a novel framework capable of generating visually appealing commodity images from textual prompts, with optional…»

*Эталон:* Датасет из 400k e-commerce примеров (изображение, двуязычный промпт, маска, референс фона). Шаги: LLM строит промпты, Seedream3 генерирует изображения, GroundingDINO + SAM дают маски, BAGEL удаляет товар и строит референс фона.

*Ключевые факты:* 0.33 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.64 с

## Q10. На какой базовой модели построен DreamPainter и почему выбрана именно она?

DreamPainter построен на базе **CogView4** [1].
Выбор обоснован подходящим размером модели и поддержкой двух языков (китайского и английского), что важно для задач с мультимодальным кондиционированием (текст + визуальные референсы) [1].

**Источники:**

- [1] DreamPainter, 5 Experiments > 5.1 Experimental Setups, стр. 6 — «**Trainging Details.** DreamPainter is built upon CogView4 [35, 34], for its appropriate model size and dual-language (Chinese and English) support.»

*Эталон:* На CogView4 — из-за подходящего размера модели и поддержки двух языков (китайского и английского).

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.01 с

## Q11. Как устроено двухэтапное обучение DreamPainter?

Двухэтапное обучение в **DreamPainter** устроено так:

1. **Этап 1 (текстовый контроль)**:
   Обучение ведётся только по текстовым промптам. Оптимируются параметры отдельной *контрольной ветви* (6 слоёв DIT Blocks), остальные параметры заморожены. Цель — научить модель выполнять фоновую инпайтинг по тексту [4].

2. **Этап 2 (адаптация к референс-изображениям)**:
   Замораживаются параметры контрольной ветви и базовой модели. Вводится **LoRA** (ранг 256) во все линейные слои основной ветви для обработки информации из референс-изображений. Референс-изображения включаются с вероятностью 50% и случайно изменяются по размеру (512×512 или 800×800) для обучения модели к их использованию [2, 4].

Ключевые моменты: на этапе 2 модель обучается работать как с текстом, так и с референс-изображениями, с балансировкой за счёт случайного отбрасывания референса [4].

**Источники:**

- [2] DreamPainter, 5 Experiments > 5.1 Experimental Setups, стр. 6 — «The training process was composed of two stages: (1) **Stage 1** : We introduced a control branch comprising 6 layers of DIT Blocks, fine-tuning only this branch while keeping other parameters frozen.»
- [4] DreamPainter, 4 Methodology > 4.2 Two-Stage Training Strategy, стр. 5, 6 — «**Stage 2: Reference-aware adaptation.** In the second stage, we freeze the control branch and original model parameters, then introduce learnable LoRA into the DiT blocks.»

*Эталон:* Этап 1: обучается только control branch (6 DiT-блоков) с текстовым управлением, ~1k GPU-часов. Этап 2: control branch и база заморожены, добавляется LoRA ранга 256, референсы подаются в 50% примеров, ~700 GPU-часов.

*Ключевые факты:* 0.67 · *цитаты из нужной статьи:* 1.00 · *генерация:* 3.14 с

## Q12. Чем LoRA scaling отличается от attention modulation в DreamPainter при малых значениях масштаба?

LoRA scaling при малых значениях (менее 0.4) приводит к **резкому ухудшению качества** изображений, так как модель возвращается к состоянию после первой стадии обучения (T2I inpainting) и не распознаёт входные паттерны ссылочного изображения [1].

Attention modulation же позволяет **увеличивать случайность** без потери качества даже при значениях масштаба ниже 0.4, так как модулирует только внимание между целевыми и ссылочными токенами [1, 2].

**Источники:**

- [1] DreamPainter, 5 Experiments > 5.4 Analyses, стр. 10 — «To assess those methods, we conducted the following experiments: (1) Fixing the LoRA scale at 1.0, we set the Attention scale to 0.8, 0.6, 0.4, 0.2, and 0.0 respectively; (2) Fixing the Attention scale at 1.0, we set the LoRA scale to 0.8,…»
- [2] DreamPainter, 4 Methodology > 4.3 Flexible Control of Reference Images, стр. 6 — «**LoRA scaling.** Since the LoRA parameters are optimized specifically for reference integration in stage 2, we can scale their output to modulate reference influence: where _ϕ_ ( · ) denotes the linear layer in original DiT blocks, _s_ is…»

*Эталон:* При масштабе ≥ 0.4 оба способа регулируют случайность; ниже 0.4 attention modulation продолжает работать без потери качества, а уменьшение LoRA scale резко ухудшает изображения: модель откатывается к состоянию после первого этапа и не понимает формат референса.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.67 с

## Q13. Из каких этапов состоит синтетическая аугментация данных в работе Preserving Product Fidelity?

Синтетическая аугментация данных в работе состоит из трёх этапов:
1. **Novel View Generation** — генерация новых ракурсов с помощью image-to-video diffusion моделей [1];
2. **New Context Images** — изменение фона с использованием masked outpainting и новых контекстных промптов [1];
3. **Включение отрицательных примеров** (не упомянуто в заголовках, но видно из [1] и [4a]) — создание в-классовых негативов для обучения модели различать продукты [1, 4a].

**Источники:**

- [1] Preserving Product Fidelity, 2 METHOD > 2.1 SYNTHETIC DATA AUGMENTATION, стр. 2, 3 — «#### 2.1.2 NEW CONTEXT IMAGES To reduce overfitting to the background elements in input images, we used masked outpainting to change the backgrounds of the base, and novel view images.»

*Эталон:* Генерация новых ракурсов через image-to-video диффузию, смена фона маскированным outpainting (маска сегментации + промпты от LLM) и добавление негативов/контрфактуалов.

*Ключевые факты:* 0.67 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.63 с

## Q14. Какое соотношение позитивных и негативных примеров использовали в Preserving Product Fidelity и что такое контрфактуалы?

В статье использовалось соотношение **2:1 позитивных к негативным примерам** [1].
Контрфактуалы — это **изображения с тем же фоном, что и у оригинального продукта, но с другими объектами, вставленными в маскированную область продукта** [1]. Они создаются для улучшения рендеринга и сохранения приоритетов классов.

**Источники:**

- [1] Preserving Product Fidelity, 2 METHOD > 2.1 SYNTHETIC DATA AUGMENTATION, стр. 3 — «We observed this gave the following benefits: (1) preserve class priors in the base generation model, (2) accurately render non-product objects in the background/foreground (3) render products with reduced artifacts in the image.»

*Эталон:* 2:1 (позитивы:негативы) — выше качество и точность ценой разнообразия. Контрфактуалы — изображения с тем же фоном, где в маску товара вписаны другие объекты.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.16 с

## Q15. Какие результаты человеческой оценки получил метод Preserving Product Fidelity по сравнению с DreamBooth + LoRA?

По результатам человеческой оценки метод **Preserving Product Fidelity** показал **17.40% проходов по изображению** и **45.5% проходов по продукту**, в то время как **DreamBooth + LoRA** имел **10.00%** и **24.00%** соответственно ([3]).

Метод автора превзошёл базовый подход по обоим показателям ([3]).

**Источники:**

- [3] Preserving Product Fidelity, 3 RESULTS > 3.2 QUANTITATIVE RESULTS, стр. 4 — «|Dataset|Overall Per Image pass rate↑|Per Product pass rate↑| |---|---|---| |Furniture (Ours)|**17.40%**|**45.5%**| |Furniture (DreamBooth)|10.00%|24.00%| Table 1: Human Evaluation results on multiple datasets created by mixing several…»

*Эталон:* Доля прошедших изображений 17.4% против 10%, доля товаров хотя бы с одним прошедшим изображением 45.5% против 24%; изображение проходит, если все 8 вопросов получили «да» по большинству из 3 оценщиков.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.34 с

## Q16. Как ранг LoRA (1 против 64) влияет на точность товара и разнообразие сцен в Preserving Product Fidelity?

При использовании **LoRA rank 1** модель быстрее достигает высокой точности (продукт-файделити) деталей товара, но уже на **700 шагах обучения** начинает переобучаться, теряя разнообразие сцен и даже воспроизводя обучающие данные при изменении промпта. При этом она генерирует более простые и однообразные сцены [2].

При **LoRA rank 64** модель дольше обучается на детализацию товара (достигает высокой точности только к **1800 шагам**), но сохраняет способность генерировать разнообразные и реалистичные сцены с уникальными ракурсами и сложными композициями даже после обучения [1, 2]. Однако и в этом случае с ростом шагов обучения разнообразие сцен постепенно снижается [2].

**Источники:**

- [1] Preserving Product Fidelity, A APPENDIX / SUPPLEMENTAL MATERIAL > A.5 LORA RANK ABLATION QUALITATIVE RESULTS, стр. 11, 12 — «LoRA rank 1 achieves high product fidelity in much fewer training steps but generates less diverse scenes, before eventually overfitting and beginning to reproduce training data regardless of prompt.»
- [2] Preserving Product Fidelity, A APPENDIX / SUPPLEMENTAL MATERIAL > A.5 LORA RANK ABLATION QUALITATIVE RESULTS, стр. 11 — «The rank 1 model learned the appearance of the desk much quicker than rank 64 model (the samples display high product fidelity at 700 steps as opposed to 1800 steps 8).»

*Эталон:* Ранг 1 выучивает товар быстрее (700 шагов против 1800), но генерирует простые сцены и переобучается; ранг 64 учится дольше, зато сохраняет разнообразие и сложность сцен.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.44 с

## Q17. В каких из работ используется LoRA и для чего?

LoRA используется в следующих работах из контекста:

1. **Preserving Product Fidelity** ([1], [3], [5]):
   - Используется для **fine-tuning** модели с целью сохранения точности воспроизведения продуктов (product fidelity).
   - В работе ([5]) отмечается, что LoRA-finetuning даёт лучшие результаты по сравнению с полным finetuning модели.
   - Исследуется зависимость от **rank LoRA** ([1], [3]): низкий ранг (например, 1) быстро улучшает точность, но приводит к переобучению и потере разнообразия сцен; высокий ранг (например, 64) сохраняет разнообразие, но требует больше шагов обучения.

2. **DreamPainter** ([2], [4]):
   - LoRA применяется для **интеграции и модуляции влияния ссылочных изображений** в процессе генерации.
   - Вводится **LoRA scaling** ([4]) для гибкого контроля над степенью соответствия ссылочному изображению и случайности в генерации.
   - Исследуется взаимодействие LoRA с **attention modulation** ([2]), чтобы балансировать между сохранением деталей и разнообразием.

**Источники:**

- [1] Preserving Product Fidelity, A APPENDIX / SUPPLEMENTAL MATERIAL > A.5 LORA RANK ABLATION QUALITATIVE RESULTS, стр. 11, 12 — «LoRA rank 1 achieves high product fidelity in much fewer training steps but generates less diverse scenes, before eventually overfitting and beginning to reproduce training data regardless of prompt.»
- [2] DreamPainter, 5 Experiments > 5.4 Analyses, стр. 10 — «To assess those methods, we conducted the following experiments: (1) Fixing the LoRA scale at 1.0, we set the Attention scale to 0.8, 0.6, 0.4, 0.2, and 0.0 respectively; (2) Fixing the Attention scale at 1.0, we set the LoRA scale to 0.8,…»
- [3] Preserving Product Fidelity, A APPENDIX / SUPPLEMENTAL MATERIAL > A.5 LORA RANK ABLATION QUALITATIVE RESULTS, стр. 11 — «The rank 1 model learned the appearance of the desk much quicker than rank 64 model (the samples display high product fidelity at 700 steps as opposed to 1800 steps 8).»
- [4] DreamPainter, 4 Methodology > 4.3 Flexible Control of Reference Images, стр. 6 — «**LoRA scaling.** Since the LoRA parameters are optimized specifically for reference integration in stage 2, we can scale their output to modulate reference influence: where _ϕ_ ( · ) denotes the linear layer in original DiT blocks, _s_ is…»
- [5] Preserving Product Fidelity, 2 METHOD > 2.4 MODEL FINETUNING, стр. 3 — «We chose to adopt the approach described in (Ruiz et al., 2023), with three adjustments: Class prior preservation loss is omitted in favor of more negatives during finetuning, we trained on a higher learning rate and for longer and we…»

*Эталон:* DreamPainter — LoRA ранга 256 на втором этапе для учёта референсного изображения (её масштаб регулирует влияние референса); Preserving Product Fidelity — LoRA-дообучение в стиле DreamBooth под конкретный товар.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 3.25 с

## Q18. Какие работы используют image-to-video или видео-диффузионные модели и с какой целью?

В предоставленных фрагментах упоминаются следующие работы, использующие **image-to-video** или **видео-диффузионные модели**:

1. **ViewCrafter** ([1], [4], [5]) использует **image-to-video diffusion model DynamiCrafter** для генерации динамических видео из одного входного изображения. Цель — синтез новых ракурсов (*novel views*) для сцен из скудных или одиночных входных данных.

2. Работа из статьи **"Preserving Product Fidelity"** ([2], [3]) применяет **image-to-video diffusion** в рамках пайплайна аугментации данных для создания синтетических тренировочных образцов. Это помогает преодолеть ограничения реальных данных при реконтекстуализации изображений продуктов.

В обоих случаях видео-диффузионные модели используются для улучшения качества и разнообразия генерации изображений/видео.

**Источники:**

- [1] ViewCrafter, 3 METHOD > 3.1 Preliminary: Video Diffusion Models, стр. 4 — «In this work, we build our model based on an open-sourced Image-to-Video (I2V) diffusion model DynamiCrafter [18], which is capable of creating dynamic videos from a single input image.»
- [2] Preserving Product Fidelity, ABSTRACT, стр. 1, 2 — «We present a framework for high-fidelity product image recontextualization using text-to-image diffusion models and a novel data augmentation pipeline.»
- [3] Preserving Product Fidelity, A APPENDIX / SUPPLEMENTAL MATERIAL > A.1 SYNTHETIC AUGMENTATION TECHNIQUES, стр. 5, 8, 9 — «Figure 2: Novel view generated using image-to-video diffusion.»
- [4] ViewCrafter, 1 INTRODUCTION, стр. 1 — «With these in mind, we propose integrating the generative power of video diffusion models with the coarse 3D prior provided by point-based representation, aiming to facilitate higher-fidelity novel view synthesis of generic scenes.»
- [5] ViewCrafter, 3 METHOD, стр. 3 — «In the following, we start with a brief introduction to video diffusion models in Section.»

*Эталон:* ViewCrafter — видео-диффузия, обусловленная рендерами облака точек, для синтеза новых ракурсов сцены; Preserving Product Fidelity — image-to-video диффузия для генерации новых ракурсов товара в обучающих данных.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.36 с

## Вне корпуса: Какой FID получает Stable Diffusion 3 на MS-COCO?

В предоставленных фрагментах нет ответа на этот вопрос.

*Отказ:* да

## Вне корпуса: Сколько параметров у модели Sora и на каких данных её обучали?

В предоставленных фрагментах нет информации о модели **Sora** или её параметрах, а также о данных, на которых её обучали.

*Отказ:* да

