"""
Unit-тесты для менеджера соединений SqliteConnectionManager.
Проверяет маршрутизацию соединений между потоками, управление транзакциями,
auto_commit, vacuum и корректное закрытие ресурсов.
"""

import threading
import pytest
from pathlib import Path
from knowledge_mcp.infrastructure.db.connection_pool import SqliteConnectionManager


def test_connection_manager_basic_lifecycle(tmp_path: Path):
    """Проверка создания, выполнения базовых операций и закрытия пула."""
    db_file = tmp_path / "test_pool.db"
    manager = SqliteConnectionManager(db_file)

    assert not manager.read_only
    assert not manager.in_transaction
    assert manager.writer_conn is not None

    conn = manager.conn
    assert conn is manager.writer_conn

    # Создание таблицы через writer
    conn.execute("CREATE TABLE test_data (id INTEGER PRIMARY KEY, val TEXT);")
    conn.execute("INSERT INTO test_data (val) VALUES ('item1');")
    manager.auto_commit()

    # Чтение
    row = conn.execute("SELECT val FROM test_data WHERE id = 1;").fetchone()
    assert row[0] == "item1"

    # Проверка vacuum_mode
    v_mode = manager.get_vacuum_mode()
    assert isinstance(v_mode, int)

    # VACUUM и PRAGMA optimize
    manager.optimize()
    manager.vacuum()

    manager.close()


def test_connection_manager_transactions(tmp_path: Path):
    """Проверка явных транзакций (begin, commit, rollback)."""
    db_file = tmp_path / "test_tx.db"
    manager = SqliteConnectionManager(db_file)

    conn = manager.conn
    conn.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance INTEGER);")
    manager.auto_commit()

    # 1. Commit транзакции
    manager.begin_transaction()
    assert manager.in_transaction is True
    conn.execute("INSERT INTO accounts (id, balance) VALUES (1, 100);")
    manager.auto_commit()  # Не должен вызывать commit во время явной транзакции
    manager.commit_transaction()
    assert manager.in_transaction is False

    res = conn.execute("SELECT balance FROM accounts WHERE id = 1;").fetchone()
    assert res[0] == 100

    # 2. Rollback транзакции
    manager.begin_transaction()
    conn.execute("INSERT INTO accounts (id, balance) VALUES (2, 200);")
    manager.rollback_transaction()
    assert manager.in_transaction is False

    res2 = conn.execute("SELECT balance FROM accounts WHERE id = 2;").fetchone()
    assert res2 is None

    # Повторный rollback / commit вне транзакции не должен падать
    manager.rollback_transaction()
    manager.commit_transaction()

    manager.close()


def test_connection_manager_multithreading(tmp_path: Path):
    """Проверка изоляции читающих потоков и потокобезопасности."""
    db_file = tmp_path / "test_mt.db"
    manager = SqliteConnectionManager(db_file)

    # Инициализация схемы
    manager.conn.execute("CREATE TABLE counter (id INTEGER PRIMARY KEY, cnt INTEGER);")
    manager.conn.execute("INSERT INTO counter (id, cnt) VALUES (1, 42);")
    manager.auto_commit()

    reader_results = []
    reader_conns = []

    def reader_worker():
        # В другом потоке менеджер должен вернуть отдельный thread-local коннект
        r_conn = manager.conn
        reader_conns.append(r_conn)
        row = r_conn.execute("SELECT cnt FROM counter WHERE id = 1;").fetchone()
        reader_results.append(row[0])

    t = threading.Thread(target=reader_worker)
    t.start()
    t.join()

    assert reader_results == [42]
    # Убеждаемся, что поток-читатель получил не writer_conn
    assert reader_conns[0] is not manager.writer_conn

    # Смена writer thread id
    orig_writer_id = threading.get_ident()
    manager.register_writer_thread(999999)
    # Теперь текущий поток считается читателем, возвращается local conn
    thread_read_conn = manager.conn
    assert thread_read_conn is not manager.writer_conn

    # Возврат обратно
    manager.register_writer_thread(orig_writer_id)
    assert manager.conn is manager.writer_conn

    manager.close()


def test_connection_manager_read_only_mode(tmp_path: Path):
    """Проверка работы менеджера в режиме read_only."""
    db_file = tmp_path / "test_ro.db"
    # Сначала создаем БД
    init_mgr = SqliteConnectionManager(db_file)
    init_mgr.conn.execute("CREATE TABLE ro_table (id INTEGER PRIMARY KEY);")
    init_mgr.auto_commit()
    init_mgr.close()

    # Открываем в read_only
    ro_mgr = SqliteConnectionManager(db_file, read_only=True)
    assert ro_mgr.read_only is True

    # Чтение доступно
    rows = ro_mgr.conn.execute("SELECT * FROM ro_table;").fetchall()
    assert len(rows) == 0

    ro_mgr.close()
