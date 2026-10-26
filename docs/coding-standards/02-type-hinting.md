# 02. Type Hinting Standards

## Mandatory Type Annotations
All public functions, methods, and properties must declare complete signatures with explicit type annotations for all parameters and return values.

### Non-Compliant:
```python
def process_webhook(payload):
    return payload.get("id")
```

### Compliant:
```python
def process_webhook(payload: dict[str, Any]) -> str | None:
    return payload.get("id")
```

## Python 3.10+ Native Syntax
Because the project targets modern Python (>= 3.11), use modern native syntax for type unions and standard collections rather than legacy `typing` constructs:
* Use `|` instead of `Union` and `Optional`:
  * `Optional[str]` -> `str | None`
  * `Union[int, float]` -> `int | float`
* Use built-in generic types for collections:
  * `typing.List[int]` -> `list[int]`
  * `typing.Dict[str, Any]` -> `dict[str, Any]`
  * `typing.Tuple[str, int]` -> `tuple[str, int]`

## Usage of `Any` and `type: ignore`
1. Avoid `Any` whenever possible. If an object is dynamic or heterogeneous, prefer `typing.Protocol`, generic variables, or typed dictionaries (`dict[str, Any]`).
2. If `# type: ignore` is required to resolve third-party library typing limitations, always append an explanatory comment:
   ```python
   session = client.get_session()  # type: ignore # Library lacks stubs for dynamic sessions
   ```
