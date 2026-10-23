# Ответы RAG (markdown-c800-o100-h-hybrid-bge-m3, top-5, перевод запроса)

| Метрика | Значение |
|---|---|
| ключевые факты в ответе | 0.83 |
| ответы с цитатами | 1.00 |
| цитаты из нужной статьи | 1.00 |
| среди цитат есть фрагмент-доказательство | 0.94 |
| отказ на вопросах вне корпуса | 2/2 |
| генерация, с (среднее) | 2.1 |
| токенов промпта (среднее) | 2286 |

## Q01. Из каких двух основных компонентов состоит Direct3D и за что отвечает каждый из них?

Direct3D состоит из двух основных компонентов:
1. **D3D-VAE** (Direct 3D Variational Auto-Encoder) — отвечает за эффективное кодирование высокоразрешенных 3D форм в компактное и непрерывное пространство латентных трипланов, а также за декодирование геометрии с использованием полунепрерывной стратегии выборки поверхности [1].
2. **D3D-DiT** (Direct 3D Diffusion Transformer) — моделирует распределение закодированных 3D латентов и предназначен для объединения позиционной информации из трёх триплановых карт признаков, обеспечивая native 3D генеративную модель, масштабируемую на большие 3D датасеты [1].

**Источники:**

- [1] Direct3D, Abstract, стр. 1 — «Our approach comprises two primary components: a Direct 3D Variational Auto-Encoder ( **D3D-VAE** ) and a Direct 3D Diffusion Transformer ( **D3D-DiT** ).»

*Эталон:* D3D-VAE кодирует 3D-форму (облако точек) в компактное латентное триплановое пространство; D3D-DiT — диффузионный трансформер, генерирующий латенты в этом пространстве по входному изображению.

*Ключевые факты:* 0.67 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.29 с

## Q02. Что такое semi-continuous occupancy в D3D-VAE и зачем она нужна?

**Semi-continuous occupancy** в D3D-VAE — это подход, при котором occupancy (наличие материала в точке) задаётся не только как дискретное значение (0 или 1), а также как **континуальное значение между 0 и 1** для точек, находящихся близко к поверхности объекта (если расстояние до поверхности меньше порога *s* = 5121).

Этот метод вводится для **сглаживания градиентов** при обучении, так как дискретные значения occupancy могут вызывать резкие изменения градиентов у поверхности, мешая оптимизации. Формула использует **Signed Distance Function (SDF)**, чтобы плавно переходить от 0 до 1 вблизи поверхности [1].

В фрагменте [2] также отмечается, что без этого подхода реконструкция тонких структур ухудшается из-за проблем с оптимизацией.

**Источники:**

- [1] Direct3D, 3 Methods > 3.1 Direct 3D Variational Auto-Encoder, стр. 5, 6 — «The formula for the semi-continuous occupancy _o_ ( **x** ) is as follows: where _sdf_ ( **x** ) denotes the Signed Distance Function (SDF) value of **x** .»
- [2] Direct3D, A Appendix > A.1 Ablation Studies, стр. 10 — «**Semi-continuous surface sampling strategy.** We adopt a semi-continuous surface sampling strategy during the training of D3D-VAE to alleviate the optimization difficulty caused by abrupt changes in occupancy near the object surface.»

*Эталон:* Вблизи поверхности (|sdf| < s, s = 1/512) occupancy принимает непрерывные значения от 0 до 1, дальше — бинарные 0/1; это сглаживает резкие изменения градиента у поверхности и облегчает оптимизацию.

*Ключевые факты:* 0.67 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.74 с

## Q03. Какие предобученные энкодеры изображений использует D3D-DiT для пиксельного и семантического выравнивания?

D3D-DiT использует **DINO-v2 (ViT-L/14)** для **пиксельного выравнивания** [4] и **CLIP (ViT-L/14)** для **семантического выравнивания** [1].

**Источники:**

- [1] Direct3D, 3 Methods > 3.2 Image-conditioned Direct 3D Diffusion Transformer, стр. 7 — «We employ the pre-trained CLIP [40] visual model (ViT-L/14) to extract semantic image tokens **c** s from the conditional images, and then utilize a cross-attention layer within each DiT block to facilitate the interaction between **c** s…»
- [4] Direct3D, 3 Methods > 3.2 Image-conditioned Direct 3D Diffusion Transformer, стр. 6, 7 — «We employ the pre-trained DINO-v2 [36] (ViT-L/14) as the pixel-level image encoder, which has been revealed in previous work [1] to outperform other pre-trained vision models in extracting structural information beneficial for 3D tasks.»

*Эталон:* Пиксельный уровень — DINO-v2 (ViT-L/14), токены конкатенируются с латентами в self-attention; семантический уровень — CLIP ViT-L/14 через cross-attention.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.16 с

## Q04. Какой сэмплер, сколько шагов и какой guidance scale используются при инференсе Direct3D?

При инференсе **D3D-DiT** в Direct3D используется **DDIM sampler** с **50 шагами** и **guidance scale** равным **7.5** [1].

**Источники:**

- [1] Direct3D, 4 Experiments > 4.1 Implementation Details, стр. 7, 8 — «During inference, we apply 50 steps of DDIM [48] with the guidance scale set to 7.5.»

*Эталон:* 50 шагов DDIM с guidance scale 7.5.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.02 с

## Q05. На основе какой видео-диффузионной модели построен ViewCrafter и как в неё подаются рендеры облака точек?

ViewCrafter построен на основе **video diffusion model**, архитектура которого наследует **Latent Diffusion Model (LDM)**. В модель интегрированы:
- VAE-энкодер/декодер для сжатия изображений,
- видео-деноизирующий U-Net с пространственными и временными слоями для оценки шума,
- CLIP-энкодер для понимания референс-изображения.

Рендеры облака точек подаются как **условные сигналы** в видео-деноизирующий U-Net: они кодируются через VAE-энкодер, а полученные латентные изображения конкатенируются с шумом по канальному измерению [1].

**Источники:**

- [1] ViewCrafter, 3 METHOD > 3.3 Rendering High-fidelity Novel Views with Video Diffusion Models, стр. 4 — «Motivated by the efficacy of video diffusion models [16], [17], [18] in synthesizing high-quality and consistent videos, we learn this conditional distribution by training a video diffusion model conditioned on the point cloud renders and…»

*Эталон:* На image-to-video модели DynamiCrafter; рендеры облака точек кодируются VAE-энкодером и конкатенируются с шумом по каналам, референс дополнительно идёт через CLIP и cross-attention.

*Ключевые факты:* 0.33 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.28 с

## Q06. Как ViewCrafter реконструирует облако точек по одному или нескольким изображениям?

ViewCrafter использует **dense stereo model** (например, DUSt3R) для реконструкции облака точек из одного или нескольких изображений [4].
Модель принимает на вход пару RGB-изображений (или дублирует одно изображение при отсутствии пары) и генерирует **точечные карты** (*point maps*) и **конфиденциальные карты** (*confidence maps*), отражающие точность точек в координатах камеры.
При наличии нескольких изображений выполняется **глобальная выравнивание точечных карт** для улучшения точности.
Цветное облако точек получают путем интеграции точечных карт с соответствующими RGB-изображениями.

**Источники:**

- [4] ViewCrafter, 3 METHOD > 3.2 Point Cloud Reconstruction from Single or Sparse Images, стр. 4 — «Specifically, we employ a dense stereo model, _e.g._ DUSt3R [19], to reconstruct the point cloud and estimate camera parameters simultaneously.»

*Эталон:* Плотной стерео-моделью DUSt3R: она строит point maps и карты уверенности, фокусное расстояние находится алгоритмом Вейсфельда; одно изображение дублируется в пару.

*Ключевые факты:* 0.50 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.24 с

## Q07. Как устроен алгоритм планирования траектории камеры в ViewCrafter и с какими параметрами он запускался в абляции?

Алгоритм планирования траектории камеры в ViewCrafter основан на подходе Next-Best-View (NBV) и использует поиск в четверти-сфере (forward-facing quarter-sphere) с равномерно распределёнными позициями камеры. Радиус поискового пространства равен глубине центрального пикселя в исходном изображении.

В абляции алгоритм запускался с параметрами:
- *N* = 3 (число повторений движения в каждом направлении),
- *K* = 5 (количество кандидатов для выборки),
- Θ = 0.6 (параметр для оценки полезности камеры),
- Камера перемещалась на 20 градусов за одно движение [1, 4].

Поиск начинался с левой половины пространства, затем — с правой.

**Источники:**

- [1] ViewCrafter, 4 EXPERIMENTS > 4.4 Ablation Study, стр. 11 — «We begin with exploring the left half area of the searching space, with the parameters for camera trajectory planning set to _N_ = 3, _K_ = 5, and Θ = 0.6.»
- [4] ViewCrafter, 3 METHOD > 3.4 Iterative View Synthesis and Camera Trajectory Planning, стр. 5 — «To effectively reveal occlusions in the iterative view synthesis process and facilitate more complete scene generation, we designed a Next-Best-View (NBV) [64], [65], [66] -based camera trajectory planning algorithm, which enables the…»

*Эталон:* Next-Best-View: из пространства поиска (четверть сферы) сэмплируются K кандидатных поз, по маскам дыр утилити-функция выбирает позу, открывающую достаточно, но не слишком много пропусков (порог Θ); затем генерация, обратная проекция и обновление облака. В абляции N = 3, K = 5, Θ = 0.6.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.65 с

## Q08. На каких данных и в каком объёме обучали ViewCrafter?

ViewCrafter обучали на смешанном датасете, состоящем из **DL3DV** и **RealEstate10K**.
В процессе обучения использовали **632 152 пар видео** (видео + соответствующие рендеры точечных облаков) [1].

**Источники:**

- [1] ViewCrafter, 4 EXPERIMENTS > 4.1 Implementation Details, стр. 6 — «Our model was trained on a mixed dataset consisting of DL3DV [70] and RealEstate10K [7].»

*Эталон:* На смеси DL3DV и RealEstate10K: клипы по 25 кадров, рендеры облаков точек через DUSt3R и PyTorch3D — всего 632 152 пары видео.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.22 с

## Q09. Что такое DreamEcom-400K и из каких шагов состоит пайплайн его построения?

DreamEcom-400K — это высококачественный датасет для задач генерации изображений в сфере e-commerce, содержащий 400 000 разнообразных образцов, включающих товарные изображения, двуязычные (китайский-английский) промпты, инстанс-маски и фоновые ссылочные изображения [1].

Пайплайн его построения включает следующие шаги:
1. **Создание текстовых промптов** с помощью LLM, где визуальный контент разбивается на 5 семантических компонентов (продукты, подставки, окружающая среда, фон, стиль) [1].
2. **Генерация изображений** с использованием T2I-модели Seedream3 на основе созданных промптов [1].
3. **Получение инстанс-масок** с помощью комбинации GroundingDINO и SAM [1].
4. **Создание ссылочных фоновых изображений** с помощью модели BAGLE для семантической правки [1].

**Источники:**

- [1] DreamPainter, 3 Dataset, стр. 3, 4 — «2 (right), Which includes the following steps: **Step 1: Using LLM to build image prompts.** The visual content is decomposed into five semantic components: (1) products, encompassing attributes such as categories, quantities, and colors;…»

*Эталон:* Датасет из 400k e-commerce примеров (изображение, двуязычный промпт, маска, референс фона). Шаги: LLM строит промпты, Seedream3 генерирует изображения, GroundingDINO + SAM дают маски, BAGEL удаляет товар и строит референс фона.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.94 с

## Q10. На какой базовой модели построен DreamPainter и почему выбрана именно она?

DreamPainter построен на базе **CogView4** [1].
Выбор этой модели обоснован её **подходящим размером** и **поддержкой двух языков** (китайского и английского), что важно для задач, связанных с e-commerce, где тексты могут быть на разных языках.

**Источники:**

- [1] DreamPainter, 5 Experiments > 5.1 Experimental Setups, стр. 6, 7 — «**Trainging Details.** DreamPainter is built upon CogView4 [35, 34], for its appropriate model size and dual-language (Chinese and English) support.»

*Эталон:* На CogView4 — из-за подходящего размера модели и поддержки двух языков (китайского и английского).

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.33 с

## Q11. Как устроено двухэтапное обучение DreamPainter?

Двухэтапное обучение в **DreamPainter** устроено следующим образом:

**Этап 1: Текстовый контроль (Text-only control)**
Модель обучается только на текстовых промптах, оптимируются только параметры **контрольного ветви** (control branch), чтобы научиться вписывать фон по тексту. Остальные параметры замораживаются [1].

**Этап 2: Адаптация с учетом референса (Reference-aware adaptation)**
Контрольный ветвь и основные параметры замораживаются. В **DiT-блоки** вводятся **обучаемые LoRA**, чтобы интегрировать информацию о референс-изображении. Во время обучения **случайно отбрасывается референс с высокой вероятностью** (50% по [4]), чтобы модель могла работать и без него. Также референс-изображения **случайно сдвигаются** для улучшения восприятия пространственных отношений [1].

LoRA добавляются ко всем линейным слоям основной модели [4].

**Источники:**

- [1] DreamPainter, 4 Methodology > 4.2 Two-Stage Training Strategy, стр. 5, 6 — «**Stage 2: Reference-aware adaptation.** In the second stage, we freeze the control branch and original model parameters, then introduce learnable LoRA into the DiT blocks.»
- [4] DreamPainter, 5 Experiments > 5.1 Experimental Setups, стр. 6, 7 — «The training process was composed of two stages: (1) **Stage 1** : We introduced a control branch comprising 6 layers of DIT Blocks, fine-tuning only this branch while keeping other parameters frozen.»

*Эталон:* Этап 1: обучается только control branch (6 DiT-блоков) с текстовым управлением, ~1k GPU-часов. Этап 2: control branch и база заморожены, добавляется LoRA ранга 256, референсы подаются в 50% примеров, ~700 GPU-часов.

*Ключевые факты:* 0.67 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.97 с

## Q12. Чем LoRA scaling отличается от attention modulation в DreamPainter при малых значениях масштаба?

При малых значениях масштаба (менее 0.4) **LoRA scaling** приводит к резкому ухудшению качества сгенерированных изображений из-за несоответствия между обучением и инференсом: модель возвращается к состоянию после первой стадии обучения (T2I inpainting), не распознавая входные паттерны ссылочного изображения и не генерируя разумные результаты [1].

В то же время **attention modulation** позволяет увеличивать случайность без ухудшения качества: регулировка внимания сохраняет качество, в отличие от LoRA [1].

**Источники:**

- [1] DreamPainter, 5 Experiments > 5.4 Analyses, стр. 10 — «To assess those methods, we conducted the following experiments: (1) Fixing the LoRA scale at 1.0, we set the Attention scale to 0.8, 0.6, 0.4, 0.2, and 0.0 respectively; (2) Fixing the Attention scale at 1.0, we set the LoRA scale to 0.8,…»

*Эталон:* При масштабе ≥ 0.4 оба способа регулируют случайность; ниже 0.4 attention modulation продолжает работать без потери качества, а уменьшение LoRA scale резко ухудшает изображения: модель откатывается к состоянию после первого этапа и не понимает формат референса.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.0 с

## Q13. Из каких этапов состоит синтетическая аугментация данных в работе Preserving Product Fidelity?

Синтетическая аугментация данных в работе состоит из трёх основных этапов:

1. **Novel View Generation** — генерация новых ракурсов изображений с помощью видео-диффузионных моделей (image-to-video diffusion models) для разнообразия перспектив продукта ([1]).

2. **New Context Images** — изменение фонов изображений с помощью маскирования и outpainting, с использованием новых контекстных промптов, сгенерированных LLM ([1]).

3. **Augment with Negatives** — добавление негативных примеров (включая контрфактуальные изображения с другими объектами на месте продукта) для сохранения приоритетов классов и улучшения качества рендеринга ([1]).

**Источники:**

- [1] Preserving Product Fidelity, 2 METHOD > 2.1 SYNTHETIC DATA AUGMENTATION, стр. 2, 3 — «We observed this gave the following benefits: (1) preserve class priors in the base generation model, (2) accurately render non-product objects in the background/foreground (3) render products with reduced artifacts in the image.»

*Эталон:* Генерация новых ракурсов через image-to-video диффузию, смена фона маскированным outpainting (маска сегментации + промпты от LLM) и добавление негативов/контрфактуалов.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.05 с

## Q14. Какое соотношение позитивных и негативных примеров использовали в Preserving Product Fidelity и что такое контрфактуалы?

В статье используется соотношение **2:1 позитивных к негативным примерам** [2].

Контрфактуалы (*counterfactuals*) — это **изображения с тем же фоном, что и у оригинальных (ground truth) изображений, но с другими объектами, инпайнтированными в область продукта** [2]. Они создаются для:
- сохранения классовых приоритетов в модели,
- точного рендеринга не-продуктовых объектов на фоне/первом плане,
- уменьшения артефактов на изображении продукта.

**Источники:**

- [2] Preserving Product Fidelity, 2 METHOD > 2.1 SYNTHETIC DATA AUGMENTATION, стр. 2, 3 — «We observe that a 2:1 positive:negative image ratio produced higher image quality and fidelity at the cost of diversity.»

*Эталон:* 2:1 (позитивы:негативы) — выше качество и точность ценой разнообразия. Контрфактуалы — изображения с тем же фоном, где в маску товара вписаны другие объекты.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.64 с

## Q15. Какие результаты человеческой оценки получил метод Preserving Product Fidelity по сравнению с DreamBooth + LoRA?

По результатам человеческой оценки метод **Preserving Product Fidelity** показал **17.40% проходов по изображению** и **45.5% проходов по продукту**, в то время как **DreamBooth + LoRA** получил **10.00%** и **24.00%** соответственно ([1]).

Метод автора значимо превосходит базовый подход по обоим показателям.

**Источники:**

- [1] Preserving Product Fidelity, 3 RESULTS > 3.2 QUANTITATIVE RESULTS, стр. 4, 5 — «|Dataset|Overall Per Image pass rate↑|Per Product pass rate↑| |---|---|---| |Furniture (Ours)|**17.40%**|**45.5%**| |Furniture (DreamBooth)|10.00%|24.00%| Table 1: Human Evaluation results on multiple datasets created by mixing several…»

*Эталон:* Доля прошедших изображений 17.4% против 10%, доля товаров хотя бы с одним прошедшим изображением 45.5% против 24%; изображение проходит, если все 8 вопросов получили «да» по большинству из 3 оценщиков.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 1.43 с

## Q16. Как ранг LoRA (1 против 64) влияет на точность товара и разнообразие сцен в Preserving Product Fidelity?

При использовании **LoRA rank=1** модель быстрее достигает высокой точности детализации товара (например, мебели), но при этом теряет разнообразие сцен: уже на **700 шагах обучения** начинает генерировать простые и однообразные композиции, а при дальнейшем обучении — переобучается и даже воспроизводит обучающие изображения без учета промпта [1].

При **LoRA rank=64** модель дольше обучается на детализации, но сохраняет способность генерировать более разнообразные и реалистичные сцены с уникальными ракурсами и сложными окружениями. Однако даже в этом случае с ростом шагов обучения снижается разнообразие [1].

Таким образом, **rank=1** оптимизирует точность товара, а **rank=64** — поддерживает разнообразие сцен, требуя баланса между ними [1].

**Источники:**

- [1] Preserving Product Fidelity, A APPENDIX / SUPPLEMENTAL MATERIAL > A.5 LORA RANK ABLATION QUALITATIVE RESULTS, стр. 11, 12 — «LoRA rank 1 achieves high product fidelity in much fewer training steps but generates less diverse scenes, before eventually overfitting and beginning to reproduce training data regardless of prompt.»

*Эталон:* Ранг 1 выучивает товар быстрее (700 шагов против 1800), но генерирует простые сцены и переобучается; ранг 64 учится дольше, зато сохраняет разнообразие и сложность сцен.

*Ключевые факты:* 0.67 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.64 с

## Q17. В каких из работ используется LoRA и для чего?

LoRA используется в следующих работах из контекста:

1. В статье **"Preserving Product Fidelity"** ([1], [5]) LoRA применяется для **тонкой настройки (fine-tuning)** модели. Авторы отмечают, что использование LoRA finetuning даёт лучшие результаты по сравнению с полной настройкой модели. Также в работе исследуется влияние ранга LoRA (например, rank=1 и rank=64) на баланс между точностью воспроизведения объекта и разнообразием генерируемых сцен.

2. В статье **"DreamPainter"** ([2], [3], [4]) LoRA используется для **интеграции информации из ссылочного изображения** в процесс генерации. Конкретно:
   - LoRA оптимизируется на втором этапе обучения для работы с ссылочными изображениями ([3], [4]).
   - LoRA scaling позволяет **гибко контролировать влияние ссылочного изображения** на генерацию ([2], [4]).
   - LoRA применяется в DiT блоках для адаптации модели к новому формату входных данных ([3]).

**Источники:**

- [1] Preserving Product Fidelity, A APPENDIX / SUPPLEMENTAL MATERIAL > A.5 LORA RANK ABLATION QUALITATIVE RESULTS, стр. 11, 12 — «LoRA rank 1 achieves high product fidelity in much fewer training steps but generates less diverse scenes, before eventually overfitting and beginning to reproduce training data regardless of prompt.»
- [2] DreamPainter, 5 Experiments > 5.4 Analyses, стр. 10 — «To assess those methods, we conducted the following experiments: (1) Fixing the LoRA scale at 1.0, we set the Attention scale to 0.8, 0.6, 0.4, 0.2, and 0.0 respectively; (2) Fixing the Attention scale at 1.0, we set the LoRA scale to 0.8,…»
- [3] DreamPainter, 4 Methodology > 4.1 System Framework, стр. 5 — «To inject reference image information into the generation process, we follow prior works [25, 28] and concatenate _C_ , _Z_ tgt_t_, and_Z_refalong the spatial dimension as input to the DiT blocks: We assign distinct positional encodings to…»
- [4] DreamPainter, 4 Methodology > 4.3 Flexible Control of Reference Images, стр. 6 — «**LoRA scaling.** Since the LoRA parameters are optimized specifically for reference integration in stage 2, we can scale their output to modulate reference influence: where _ϕ_ ( · ) denotes the linear layer in original DiT blocks, _s_ is…»
- [5] Preserving Product Fidelity, 2 METHOD > 2.4 MODEL FINETUNING, стр. 3 — «We chose to adopt the approach described in (Ruiz et al., 2023), with three adjustments: Class prior preservation loss is omitted in favor of more negatives during finetuning, we trained on a higher learning rate and for longer and we…»

*Эталон:* DreamPainter — LoRA ранга 256 на втором этапе для учёта референсного изображения (её масштаб регулирует влияние референса); Preserving Product Fidelity — LoRA-дообучение в стиле DreamBooth под конкретный товар.

*Ключевые факты:* 1.00 · *цитаты из нужной статьи:* 1.00 · *генерация:* 2.92 с

## Q18. Какие работы используют image-to-video или видео-диффузионные модели и с какой целью?

В контексте упоминаются следующие работы, использующие **image-to-video (I2V) или видео-диффузионные модели**:

1. **DynamiCrafter** ([1]) — открытая I2V-диффузионная модель, используемая в **ViewCrafter** для генерации динамических видео из одного входного изображения.
2. **AnimateDiff** ([2]) — исследует класс-условную генерацию видео, включая движение камеры, с использованием LoRA-модулей.
3. **SVD** ([2]) — также относится к видео-диффузионным моделям для генерации видео с управлением камерой.
4. **MultiDiff** ([2]) — использует видео-диффузионные модели с предварительной обработкой глубины для генерации видео с 3D-предварительными данными.

**Цели**:
- Управление движением камеры (например, **AnimateDiff**, **SVD**, **MotionCtrl**).
- Генерация видео из статичных изображений (**DynamiCrafter**).
- Улучшение контроля над 3D-представлением (например, **MultiDiff** с использованием глубины).

**Источники:**

- [1] ViewCrafter, 3 METHOD > 3.1 Preliminary: Video Diffusion Models, стр. 3, 4 — «In this work, we build our model based on an open-sourced Image-to-Video (I2V) diffusion model DynamiCrafter [18], which is capable of creating dynamic videos from a single input image.»
- [2] ViewCrafter, 2 RELATED WORK > 2.3 Conditional Video Diffusion Models, стр. 3 — «MultiDiff [55] leverage depth-based warping to produce warped images, and condition the video diffusion model on the warped images to provide explicit 3D prior.»

*Эталон:* ViewCrafter — видео-диффузия, обусловленная рендерами облака точек, для синтеза новых ракурсов сцены; Preserving Product Fidelity — image-to-video диффузия для генерации новых ракурсов товара в обучающих данных.

*Ключевые факты:* 0.50 · *цитаты из нужной статьи:* 1.00 · *генерация:* 3.13 с

## Вне корпуса: Какой FID получает Stable Diffusion 3 на MS-COCO?

В предоставленных фрагментах нет ответа на этот вопрос.

*Отказ:* да

## Вне корпуса: Сколько параметров у модели Sora и на каких данных её обучали?

В предоставленных фрагментах нет информации о модели **Sora** или её параметрах, а также о данных, на которых её обучали.

*Отказ:* да

