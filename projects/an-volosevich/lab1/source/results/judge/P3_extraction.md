# Оценка: P3_extraction

## Промпт

```
Извлеки из описания товара поля и верни ТОЛЬКО JSON-объект с ключами:
brand, model, screen_inches, ram_gb, storage_gb, battery_mah, color, price_rub, weight_g, warranty_months.
Числа — числами, если значения нет в тексте — null.

Описание:
«Новинка сезона — смартфон Realme GT 7 Pro в цвете «марсианский оранжевый». Флагман получил
AMOLED-дисплей диагональю 6,78 дюйма, 16 ГБ оперативной и 512 ГБ встроенной памяти, а также
аккумулятор на 6500 мАч с быстрой зарядкой 120 Вт. Официальная гарантия производителя — 1 год.
Цена в нашем магазине — 74 990 ₽, при оформлении предзаказа в подарок фирменный чехол.»
```
## Рубрика

- **accuracy**: значения полей соответствуют тексту (числа нормализованы: 1 год -> 12 мес.)
- **no_hallucination**: отсутствующие в тексте поля = null, ничего не додумано
- **instruction_following**: только JSON, все ключи, числа числами

Шкала каждого критерия 1–5 (5 — идеально). Дополнительно: hallucination (true, если в ответе есть утверждения, не следующие из промпта), overall 1–5 и короткий comment на русском.

## Ответы (18)

### id: 0e35da41

```
```json
{
  "brand": "Realme",
  "model": "GT 7 Pro",
  "screen_inches": 6.78,
  "ram_gb": 16,
  "storage_gb": 512,
  "battery_mah": 6500,
  "color": "марсианский orange",
  "price_rub": 74990,
  "weight_g": null,
  "warranty_months": 12
}
```
```

### id: c38dccc7

```
```json
{
  "brand": "Realme",
  "model": "GT 7 Pro",
  "screen_inches": 6.78,
  "ram_gb": 16,
  "storage_gb": 512,
  "battery_mah": 6500,
  "color": "марсианский оранжевый",
  "price_rub": 74990,
  "weight_g": null,
  "warranty_months": 12
}
```
```

### id: 8f7dbfb1

```
```json
{
  "brand": "Realme",
  "model": "GT 7 Pro",
  "screen_inches": 6.78,
  "ram_gb": 16,
  "storage_gb": 512,
  "battery_mah": 6500,
  "color": "марсианский оранжевый",
  "price_rub": 74990,
  "weight_g": null,
  "warranty_months": 12
}
```
```

### id: 19188133

```
{"brand": "Realme", "model": "GT 7 Pro", "screen_inches": 6.78, "ram_gb": 16, "storage_gb": 512, "battery_mah": 6500, "color": "марсианский оранжевый", "price_rub": 74990, "weight_g": null, "warranty_months": 12}
```

### id: b569d57c

```
```json
{
  "brand": "Realme",
  "model": "GT 7 Pro",
  "screen_inches": 6.78,
  "ram_gb": 16,
  "storage_gb": 512,
  "battery_mah": 6500,
  "color": "марсианский оранжевый",
  "price_rub": 74990,
  "weight_g": null,
  "warranty_months": 12
}
```
```

### id: d495b5d1

```
```json
{
  "brand": "Realme",
  "model": "GT 7 Pro",
  "screen_inches": 6.78,
  "ram_gb": 16,
  "storage_gb": 512,
  "battery_mah": 6500,
  "color": "марсианский оранжевый",
  "price_rub": 74990,
  "weight_g": null,
  "warranty_months": 12
}
```
```

### id: d26d9f77

```
```json
{
  "brand": "Realme",
  "model": "GT 7 Pro",
  "screen_inches": 6.78,
  "ram_gb": 16,
  "storage_gb": 512,
  "battery_mah": 6500,
  "color": "марсианский оранжевый",
  "price_rub": 74990,
  "weight_g": null,
  "warranty_months": 12
}
```
```

### id: b8ef35b7

```
Чтобы извлечь из описания товара необходимые поля и вернуть JSON-объект, я воспользуюсь регулярными выражениями.

```python
import re

description = """
Новинка сезона — смартфон Realme GT 7 Pro в цвете «марсианский оранжевый». Флагман получил
AMOLED-дисплей диагональю 6,78 дюйма, 16 ГБ оперативной и 512 ГБ встроенной памяти, а также
аккумулятор на 6500 мАч с быстрой зарядкой 120 Вт. Официальная гарантия производителя — 1 год.
Цена в нашем магазине — 74 990 ₽, при оформлении предзаказа в подарок фирменный чехол.
"""

# Регулярные выражения для извлечения данных
brand_pattern = r'Realme GT 7 Pro'
model_pattern = r'Realme GT 7 Pro'  # Искать можно в любом месте описания
screen_inches_pattern = r'\d+\.\d+ дюйма'  # Искать число с точкой и дюймы
ram_gb_pattern = r'\d+ ГБ'  # Искать число ГБ
storage_gb_pattern = r'\d+ ГБ'  # Искать число ГБ
battery_mah_pattern = r'\d+ мАч'  # Искать число мАч
color_pattern = r'"марсианский оранжевый"'  # Искать цвет
price_rub_pattern = r'\d+\.\d+ ₽'  # Искать число с точкой и ₽
weight_g_pattern = r''  # Не знаю, как извлечь вес из описания, поэтому оставлю это пустым
warranty_months_pattern = r'1 год'  # Искать число месяцев

# Поиск данных в описании
brand = re.search(brand_pattern, description).group()
model = re.search(model_pattern, description).group()
screen_inches = float(re.search(screen_inches_pattern, description).group().replace(' дюйма', ''))  # Уберем дюймы
ram_gb = int(re.search(ram_gb_pattern, description).group().replace(' ГБ', ''))  # Уберем ГБ
storage_gb = int(re.search(storage_gb_pattern, description).group().replace(' ГБ', ''))  # Уберем ГБ
battery_mah = int(re.search(battery_mah_pattern, description).group().replace(' мАч', ''))  # Уберем мАч
color = re.search(color_pattern, description).group()
price_rub = float(re.search(price_rub_pattern, description).group().replace(' ₽', ''))  # Уберем ₽
warranty_months = int(re.search(warranty_months_pattern, description).group().replace(' год', ''))  # Уберем год

# Сбор данных в JSON-объект
data = {
    'brand': brand,
    'model': model,
    'screen_inches': screen_inches,
    'ram_gb': ram_gb,
    'storage_gb': storage_gb,
    'battery_mah': battery_mah,
    'color': color,
    'price_rub': price_rub,
    'weight_g': None,  # Вес не извлечь
    'warranty_months': warranty_months
}

print(data)
```
Этот код извлекает данные о телефоне и собирает их в JSON-объект. В качестве веса указан `None`, поскольку из описания он не извлечься.
```

### id: 2fefbf53

```
```json
{
  "brand": "Realme",
  "model": "GT 7 Pro",
  "screen_inches": 6.78,
  "ram_gb": 16,
  "storage_gb": 512,
  "battery_mah": 6500,
  "color": "марсианский оранжевый",
  "price_rub": 74990,
  "weight_g": null,
  "warranty_months": 12
}
```
```

### id: 8925faf8

```
```json
{
  "brand": "Realme",
  "model": "GT 7 Pro",
  "screen_inches": 6.78,
  "ram_gb": 16,
  "storage_gb": 512,
  "battery_mah": 6500,
  "color": "марсианский оранжевый",
  "price_rub": 74990,
  "warranty_months": 12,
  "weight_g": null
}
```
```

### id: a186b99e

```
Вот JSON-объект с извлеченными полями:
```json
{
  "brand": "Realme",
  "model": "GT 7 Pro",
  "screen_inches": 6.78,
  "ram_gb": 16,
  "storage_gb": 512,
  "battery_mah": 6500,
  "color": "марсианский оранжевый",
  "price_rub": 74990,
  "weight_g": null,
  "warranty_months": 12
}
```
Примечание: В описании было указано, что гарантия производителя составляет 1 год, поэтому я приравнял это к 12 месяцам. Если вы хотите указать именно 1 год, то можно изменить значение на 12.0, но в приведенном выше ответе я указал 12.
```

### id: 10b5e27c

```
```json
{
  "brand": "Realme",
  "model": "GT 7 Pro",
  "screen_inches": 6.78,
  "ram_gb": 16,
  "storage_gb": 512,
  "battery_mah": 6500,
  "color": "марсианский оранжевый",
  "price_rub": 74990,
  "weight_g": null,
  "warranty_months": 12
}
```

Примечание: В описании нет информации о весе смартфона, поэтому я поставил null в поле weight_g. Аналогично, в описании нет информации о гарантии в месяцах, но есть информация о гарантии в годах, поэтому я поставил 12 месяцев (поскольку 1 год = 12 месяцев).
```

### id: 1eda4051

```
```json
{
  "brand": "Realme",
  "model": "GT 7 Pro",
  "screen_inches": 6.78,
  "ram_gb": 16,
  "storage_gb": 512,
  "battery_mah": 6500,
  "color": "марсианский оранжевый",
  "price_rub": 74990,
  "weight_g": null,
  "warranty_months": 12
}
```
```

### id: ab5ac652

```
```json
{
  "brand": "Realme",
  "model": "GT 7 Pro",
  "screen_inches": 6.78,
  "ram_gb": 16,
  "storage_gb": 512,
  "battery_mah": 6500,
  "color": "марсианский оранжевый",
  "price_rub": 74990,
  "weight_g": null,
  "warranty_months": 12
}
```

Примечание: В описании нет информации о весе смартфона, поэтому я поставил null в поле weight_g. Аналогично, в описании нет информации о гарантии в месяцах, но есть информация о гарантии в годах, поэтому я поставил 12 месяцев (поскольку 1 год = 12 месяцев).
```

### id: 5ed709b4

```
{"brand":"Realme","model":"GT 7 Pro","screen_inches":6.78,"ram_gb":16,"storage_gb":512,"battery_mah":6500,"color":"марсианский оранжевый","price_rub":74990,"weight_g":null,"warranty_months":12}
```

### id: 2b549db0

```
```json
{
  "brand": "Realme",
  "model": "GT 7 Pro",
  "screen_inches": 6.78,
  "ram_gb": 16,
  "storage_gb": 512,
  "battery_mah": 6500,
  "color": "марсианский оранжевый",
  "price_rub": 74990,
  "weight_g": null,
  "warranty_months": 12
}
```
```

### id: 787fd652

```
```json
{
  "brand": "Realme",
  "model": "GT 7 Pro",
  "screen_inches": 6.78,
  "ram_gb": 16,
  "storage_gb": 512,
  "battery_mah": 6500,
  "color": "марсианский оранжевый",
  "price_rub": 74990,
  "weight_g": null,
  "warranty_months": 12
}
```

Примечание: В описании нет информации о весе смартфона, поэтому я поставил null в поле weight_g. Аналогично, в описании нет информации о гарантии в месяцах, но есть информация о гарантии в годах, поэтому я поставил 12 месяцев (поскольку 1 год = 12 месяцев).
```

### id: 58aef33e

```
{"brand":"Realme","model":"GT 7 Pro","screen_inches":6.78,"ram_gb":16,"storage_gb":512,"battery_mah":6500,"color":"марсианский оранжевый","price_rub":74990,"weight_g":null,"warranty_months":12}
```
