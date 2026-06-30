# 06. Testing & Quality Standards

## pytest фреймворк
Для написания и запуска тестов используется `pytest`.

## Структура Arrange-Act-Assert (AAA)
Все юнит-тесты должны структурироваться по паттерну AAA:
* **Arrange:** подготовка тестового окружения, моков, фикстур и данных.
* **Act:** вызов тестируемого метода или функции.
* **Assert:** проверка результатов работы и состояния системы.

### Пример:
```python
def test_webhook_signature_validation_success():
    # Arrange
    payload = b'{"repo_id": "test"}'
    secret = "my-secret"
    # Рассчитаем верную подпись
    valid_signature = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    
    # Act
    is_valid = verify_signature(payload, secret, valid_signature)
    
    # Assert
    assert is_valid is True
```

## Использование фикстур и моков (Mocks & Fixtures)
1. Тесты не должны изменять состояние реальной кодовой базы или делать сетевые запросы.
2. Взаимодействие с Git CLI или HTTP-запросы к OmniRoute должны мокаться с помощью `unittest.mock` (или плагина `pytest-mock`):
   ```python
   def test_git_manager_clone(mocker):
       # Перехватываем вызов subprocess.run
       mock_run = mocker.patch("subprocess.run")
       
       # Запуск клонирования
       clone_repo("https://github.com/org/repo", "/tmp/dest")
       
       # Проверяем, что команда git была вызвана корректно
       mock_run.assert_called_once()
       args = mock_run.call_args[0][0]
       assert "clone" in args
   ```
3. Для тестов базы данных используйте InMemory SQLite соединение (`:memory:`) или временный файл БД, удаляемый после завершения теста.
