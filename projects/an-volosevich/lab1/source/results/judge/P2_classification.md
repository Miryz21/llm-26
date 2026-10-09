# Оценка: P2_classification

## Промпт

```
Классифицируй каждое обращение клиента ровно в одну категорию из списка ["Billing", "Tech support", "Sales"].
Ответь ТОЛЬКО JSON-массивом меток в том же порядке, без пояснений.

Обращения:
1. С карты списали деньги дважды за одну подписку, верните, пожалуйста, лишнее.
2. Приложение вылетает при попытке открыть раздел «Отчёты» после обновления.
3. Сколько будет стоить корпоративная лицензия на 50 сотрудников?
4. Не могу войти в аккаунт: пишет «неверный токен», хотя пароль правильный.
5. Пришлите, пожалуйста, закрывающие документы и счёт-фактуру за сентябрь.
6. Хотим перейти на тариф Enterprise, можно ли получить скидку при оплате за год?
7. Почему в счёте за октябрь появилась строка «доп. услуги», мы их не подключали?
8. Интеграция с 1С перестала выгружать заказы, в логах ошибка 502.
9. Есть ли у вас партнёрская программа для интеграторов и какие условия?
```
## Рубрика

- **correctness**: доля верных меток (сверь с заданием сам, без подсказок)
- **instruction_following**: только JSON-массив, допустимые метки, порядок и количество сохранены

Шкала каждого критерия 1–5 (5 — идеально). Дополнительно: hallucination (true, если в ответе есть утверждения, не следующие из промпта), overall 1–5 и короткий comment на русском.

## Ответы (18)

### id: 6b12741b

```
```json
[
    "Billing",
    "Tech support",
    "Sales",
    "Tech support",
    "Billing",
    "Sales",
    "Billing",
    "Tech support",
    "Sales"
]
```
```

### id: d74f1ed6

```
["Billing","Tech support","Sales","Tech support","Billing","Sales","Billing","Tech support","Sales"]
```

### id: 106821fa

```
```json
[
    "Billing",
    "Tech support",
    "Sales",
    "Tech support",
    "Billing",
    "Sales",
    "Billing",
    "Tech support",
    "Sales"
]
```
```

### id: 815ed673

```
```json
[
    "Billing",
    "Tech support",
    "Sales",
    "Tech support",
    "Billing",
    "Sales",
    "Billing",
    "Tech support",
    "Sales"
]
```
```

### id: 77a16f35

```
["Billing","Tech support","Sales","Tech support","Billing","Sales","Billing","Tech support","Sales"]
```

### id: 66b7e084

```
[
  "Billing",
  "Tech support",
  "Sales",
  "Tech support",
  "Billing",
  "Sales",
  "Billing",
  "Tech support",
  "Sales"
]
```

### id: aa284b71

```
["Billing","Tech support","Sales","Tech support","Billing","Sales","Billing","Tech support","Sales"]
```

### id: d3fb1f80

```
["Billing", "Tech support", "Sales", "Billing", "Tech support", "Sales", "Billing", "Sales", "Tech support", "Sales"]
```

### id: 8b7804a4

```
["Billing","Tech support","Sales","Tech support","Billing","Sales","Billing","Tech support","Sales"]
```

### id: 26804d5a

```
["Billing", "Tech support", "Sales", "Billing", "Tech support", "Sales", "Billing", "Sales", "Tech support", "Sales"]
```

### id: fbfd97be

```
["Billing", "Tech support", "Sales", "Billing", "Tech support", "Sales", "Billing", "Tech support", "Sales", "Tech support", "Sales"]
```

### id: ee1a23a3

```
["Billing", "Tech support", "Sales", "Billing", "Tech support", "Sales", "Billing", "Sales", "Tech support", "Sales"]
```

### id: b35694a0

```
```json
[
    "Billing",
    "Tech support",
    "Sales",
    "Tech support",
    "Billing",
    "Sales",
    "Billing",
    "Tech support",
    "Sales"
]
```
```

### id: 2b4c74a5

```
```json
[
    "Billing",
    "Tech support",
    "Sales",
    "Tech support",
    "Billing",
    "Sales",
    "Billing",
    "Tech support",
    "Sales"
]
```
```

### id: f9e1ae35

```
["Billing", "Tech support", "Sales", "Billing", "Tech support", "Sales", "Billing", "Sales", "Tech support", "Sales"]
```

### id: 820afe56

```
["Billing", "Tech support", "Sales", "Tech support", "Billing", "Sales", "Billing", "Tech support", "Sales"]
```

### id: a111043d

```
```json
[
    "Billing",
    "Tech support",
    "Sales",
    "Tech support",
    "Billing",
    "Sales",
    "Billing",
    "Tech support",
    "Sales"
]
```
```

### id: 75590c57

```
["Billing","Tech support","Sales","Tech support","Billing","Sales","Billing","Tech support","Sales"]
```
