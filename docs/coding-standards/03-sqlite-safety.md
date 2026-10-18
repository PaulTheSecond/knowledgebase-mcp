# 03. SQLite Concurrency & Safety Standards

## WAL (Write-Ahead Logging) режим
Так как база данных SQLite используется параллельно процессами Web Daemon (запись/чтение) и MCP-сервером (только чтение в IDE), принудительно устанавливайте параметры соединения:
```python
def configure_connection(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
```
Режим WAL гарантирует, что писатели не будут блокировать читателей, а читатели — писателей.

## Защита от SQL-инъекций (SQL Injection Protection)
1. **Категорически запрещено** формировать SQL-запросы с помощью конкатенации строк или f-строк:
   ```python
   # ПЛОХО! КРИТИЧЕСКАЯ УЯЗВИМОСТЬ!
   cursor.execute(f"SELECT * FROM files WHERE path = '{user_input}';")
   ```
2. Разрешено передавать переменные только через стандартные плейсхолдеры SQLite (`?`):
   ```python
   # ХОРОШО
   cursor.execute("SELECT * FROM files WHERE path = ?;", (user_input,))
   ```

## Конкурентный доступ на запись и транзакции
1. Все операции записи (INSERT, UPDATE, DELETE) должны происходить внутри явного блока транзакции с немедленной блокировкой базы, чтобы исключить риск параллельной записи из другого процесса:
   ```python
   conn.execute("BEGIN IMMEDIATE TRANSACTION;")
   ```
2. Каждое соединение должно создаваться с явным тайм-аутом блокировки базы данных (`timeout=30.0`), чтобы предотвратить падение с ошибкой `database is locked` при коротких параллельных транзакциях.
3. Соединения с базой данных должны закрываться с помощью контекстных менеджеров `with`:
   ```python
   with get_db_connection(db_path) as conn:
       cursor = conn.cursor()
       cursor.execute("...")
   ```

## Read-Only подключения для MCP
Процесс MCP-сервера, запущенный в IDE, выполняет только чтение. Во избежание блокировок и случайных изменений, подключайтесь к БД в Read-Only режиме:
```python
conn = sqlite3.connect("file:/data/knowledge.db?mode=ro", uri=True)
```

## Прагмы памяти и производительности (Production Pragmas)
Для пишущих соединений в `get_db_connection` настраиваются следующие прагмы:
```python
conn.execute("PRAGMA foreign_keys = ON;")
conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)};")
conn.execute("PRAGMA temp_store = MEMORY;")
conn.execute("PRAGMA cache_size = -64000;")  # 64 МБ кэша страниц в RAM
conn.execute("PRAGMA auto_vacuum = FULL;")
conn.execute("PRAGMA journal_mode = WAL;")
conn.execute("PRAGMA synchronous = NORMAL;")
```

## Батчевые транзакции для больших объемов данных (Batch Commit)
При циклической вставке или обновлении сотен и тысяч файлов (например, в `Indexer.sync_repo`), запрещено держать одну монолитную транзакцию на весь репозиторий.
Используйте периодический сброс батчами (рекомендуемый размер `BATCH_COMMIT_SIZE = 50`):
```python
if processed % BATCH_COMMIT_SIZE == 0:
    self.db.commit_transaction()
    self.db.begin_transaction()
```
Это предотвращает переполнение WAL-буферов и возникновение ошибок ввода-вывода (`sqlite3.OperationalError: disk I/O error`) на сетевых и Docker bind-mount томах.

