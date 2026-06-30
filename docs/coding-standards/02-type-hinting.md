# 02. Type Hinting Standards

## Обязательное использование аннотаций типов
Все функции, методы и свойства обязаны иметь полную сигнатуру с аннотациями типов входных аргументов и возвращаемого значения.

### Плохо:
```python
def process_webhook(payload):
    return payload.get("id")
```

### Хорошо:
```python
def process_webhook(payload: dict[str, Any]) -> str | None:
    return payload.get("id")
```

## Синтаксис Python 3.10+
Поскольку проект запускается на современной версии Python, используйте новый синтаксис для объединения типов и встроенных коллекций вместо импорта из модуля `typing`:
* Используйте `|` вместо `Union` и `Optional`.
  * `Optional[str]` -> `str | None`
  * `Union[int, float]` -> `int | float`
* Используйте встроенные типы для коллекций:
  * `typing.List[int]` -> `list[int]`
  * `typing.Dict[str, Any]` -> `dict[str, Any]`
  * `typing.Tuple[str, int]` -> `tuple[str, int]`

## Использование `Any` и `type: ignore`
1. Избегайте использования типа `Any` без острой необходимости. Если тип аргумента неизвестен или динамический, используйте более специфичные типы, `typing.Protocol` или `dict[str, Any]`.
2. Если вы вынуждены использовать `# type: ignore` для обхода ограничений статического анализатора (например, `mypy`), обязательно добавьте краткий комментарий, объясняющий причину:
   ```python
   session = client.get_session()  # type: ignore # Library lacks type definitions for dynamic sessions
   ```
