"""
Сквозные E2E-тесты и бенчмарки для архитектуры v1.8 (ADR-008, Эпик 8):
1. Тест 1: Обратная совместимость одиночного репозитория (Backward Compatibility):
   - Стандартный репозиторий с конфигом только repositories: (без projects:).
   - Индексация, полнотекстовый поиск, проверка ответов с дефолтными project_id="default" и branch="master".
2. Тест 2: Бенчмарк CAS-дедупликации (CAS Compression Benchmark):
   - Мульти-ворктри структура: master (50 файлов кода) и 3 задачи (orbit-1, orbit-2, orbit-3),
     в каждой изменено лишь по 2 файла.
   - Проверка ~200 логических файлов, ~56 уникальных записей в file_contents,
     сэкономленного пространства > 70%, отсутствие раздувания SQLite в 4 раза (в отличие от code-review-graph).
3. Тест 3: Надежность JIT-синхронизации (JIT Freshness Guarantee):
   - Редактирование файла напрямую на диске (имитация разработчика в Visual Studio / Rider).
   - Вызовы MCP knowledge_search и knowledge_find_symbol гарантируют, что JITSyncEngine
     обнаружил изменение, переиндексировал файл и вернул свежий код без перезапуска сервиса.
4. Тест 4: Сквозной сценарий Code Review (Blast Radius, Test Gaps, Risk Scoring):
   - Репозиторий с сервисом OrderService (метод ProcessOrder, вызываемый ApiController),
     тестом OrderServiceTests, и новой непротестированной функцией FastCheckout.
   - knowledge_review_context: проверка changed_files, blast radius (ApiController затронут),
     связанный тест OrderServiceTests найден, FastCheckout отмечен как Test Gap,
     скоринг риска (risk_score > 0, medium/high), метрика context_savings.
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

from knowledge_mcp.core.config import get_projects, load_config
from knowledge_mcp.core.indexer import Indexer
from knowledge_mcp.db.models import KnowledgeDB
import knowledge_mcp.mcp.server as mcp_server
from knowledge_mcp.mcp.server import (
    call_tool,
    knowledge_find_symbol,
    knowledge_read_file,
    knowledge_review_context,
    knowledge_search,
    set_db,
)


@pytest.fixture
def clean_env():
    """Изолированное тестовое окружение с очисткой переменных окружения и кэша MCP."""
    old_config_path = os.environ.get("CONFIG_PATH")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        db_path = tmp_root / "test_e2e_v18.db"
        db = KnowledgeDB(db_path, read_only=False)
        set_db(db)
        mcp_server.use_embeddings = False

        yield {
            "root": tmp_root,
            "db": db,
            "db_path": db_path,
        }

        db.close()
        if old_config_path is not None:
            os.environ["CONFIG_PATH"] = old_config_path
        else:
            os.environ.pop("CONFIG_PATH", None)


class TestE2EV18Benchmarks:
    """Набор сквозных тестов и бенчмарков v1.8 (Эпик 8)."""

    def test_e2e_backward_compatibility_single_repo(self, clean_env, monkeypatch):
        """
        Тест 1: Обратная совместимость одиночного репозитория (Backward Compatibility).
        - Создается стандартный репозиторий с конфигом только repositories: (без секции projects:).
        - Выполняется индексация через sync_repo.
        - Вызывается knowledge_search (напрямую и через протокол call_tool).
        - Проверяется, что ответ возвращается с дефолтными project_id="default" и branch="master",
          метрики экономии контекста (context_savings) рассчитываются корректно.
        """
        root = clean_env["root"]
        db = clean_env["db"]

        # 1. Создаем локальный репозиторий с кодом сервиса авторизации
        repo_dir = root / "legacy_auth_service"
        repo_dir.mkdir(parents=True)

        subprocess.run(["git", "init", "-b", "master"], cwd=str(repo_dir), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "QA Engineer"], cwd=str(repo_dir), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "qa@example.com"], cwd=str(repo_dir), check=True, capture_output=True)

        auth_code = '''
def authenticate_user(username: str, token: str) -> bool:
    """Аутентификация пользователя по постоянному токену доступа."""
    if not username:
        return False
    return token == "secret_legacy_token_2026"

def revoke_token(token: str) -> None:
    """Инвалидация отозванного токена."""
    pass
'''
        utils_code = '''
def extract_bearer_token(header_value: str) -> str:
    """Извлечение Bearer токена из заголовка Authorization."""
    if header_value.startswith("Bearer "):
        return header_value[7:]
    return ""
'''
        (repo_dir / "auth_manager.py").write_text(auth_code.strip(), encoding="utf-8")
        (repo_dir / "utils.py").write_text(utils_code.strip(), encoding="utf-8")

        subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit of auth service"], cwd=str(repo_dir), check=True, capture_output=True)

        # 2. Создаем config.yaml БЕЗ секции projects: (только repositories:)
        config_data = {
            "server": {
                "port": 8000,
                "host": "127.0.0.1",
                "dashboard": {"enabled": False}
            },
            "repositories": [
                {
                    "id": "legacy-auth-repo",
                    "local_path": str(repo_dir),
                    "branch": "master",
                    "language": "python"
                }
            ]
        }
        config_file = root / "config.yaml"
        config_file.write_text(yaml.dump(config_data, allow_unicode=True), encoding="utf-8")
        monkeypatch.setenv("CONFIG_PATH", str(config_file))

        # Проверяем, что секция projects: отсутствует в файле и get_projects() пуст
        raw_loaded = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        assert "projects" not in raw_loaded, "Секция projects: должна отсутствовать для проверки совместимости"
        assert get_projects() == []

        # 3. Индексируем репозиторий через стандартный пайплайн
        indexer = Indexer(db, use_embeddings=False)
        indexer.sync_repo("legacy-auth-repo", repo_dir)

        # 4. Проверяем knowledge_search без явных project_id и branch (используются значения по умолчанию)
        search_res = knowledge_search(
            query="authenticate_user",
            db=db
        )

        assert search_res["total_results"] > 0
        top_hit = search_res["results"][0]
        assert top_hit["path"] == "auth_manager.py"
        assert top_hit["repo_id"] == "legacy-auth-repo"
        assert top_hit["project_id"] == "default"
        assert top_hit["branch"] == "master"

        # Проверяем метрику контекстной экономии
        savings = search_res["context_savings"]
        assert savings["saved_tokens"] > 0
        assert savings["saved_percent"] > 0
        assert savings["baseline_tokens"] > savings["returned_tokens"]

        # 5. Проверяем вызов через MCP протокол call_tool без указания project_id и branch
        mcp_result = asyncio.run(call_tool("knowledge_search", {
            "query": "extract_bearer_token"
        }))
        assert len(mcp_result) == 1
        response_text = mcp_result[0].text
        assert "utils.py" in response_text
        assert "context_savings" in response_text

        # 6. Проверяем поиск символа knowledge_find_symbol с дефолтными параметрами
        sym_res = knowledge_find_symbol("authenticate_user", db=db)
        assert sym_res["total_symbols"] >= 1
        found_sym = sym_res["symbols"][0]
        assert found_sym["name"] == "authenticate_user"
        assert found_sym["path"] == "auth_manager.py"
        assert found_sym["repo_id"] == "legacy-auth-repo"

        # 7. Проверяем чтение файла knowledge_read_file
        file_content = knowledge_read_file("auth_manager.py", repo_id="legacy-auth-repo", db=db)
        assert file_content is not None
        assert "secret_legacy_token_2026" in file_content

    def test_e2e_cas_deduplication_compression_benchmark(self, clean_env):
        """
        Тест 2: Бенчмарк CAS-дедупликации (CAS Compression Benchmark).
        - Базовый репозиторий master содержит 50 файлов кода (~50 функций).
        - Создаются 3 параллельные ворктри задач (orbit-1, orbit-2, orbit-3).
        - В каждой ветке задачи изменено лишь по 2 файла (остальные 48 файлов идентичны master).
        - Индексируются все ворктри через CAS.
        - Валидация метрик db.get_cas_stats():
          * Общее число логических файлов: 200 (50 * 4).
          * Число уникальных записей в file_contents: 56 (50 базовых + 3 * 2 измененных).
          * Процент сэкономленного дискового пространства: > 70% (144 файла переиспользованы).
          * Чанки и символы не дублируются 4 раза в SQLite, как это происходило в code-review-graph!
        """
        root = clean_env["root"]
        db = clean_env["db"]
        indexer = Indexer(db, use_embeddings=False)

        # 1. Генерируем 50 базовых файлов для master
        master_dir = root / "worktrees" / "master"
        master_dir.mkdir(parents=True)

        base_files_map = {}
        for i in range(1, 51):
            file_name = f"service_{i:02d}.py"
            code = (
                f"# Core domain service module #{i}\n"
                f"def execute_domain_operation_{i:02d}(order_id: int) -> int:\n"
                f"    \"\"\"Выполняет типовую операцию {i:02d}.\"\"\"\n"
                f"    multiplier = {i * 10}\n"
                f"    return order_id * multiplier + 100\n"
            )
            file_path = master_dir / file_name
            file_path.write_text(code, encoding="utf-8")
            base_files_map[file_name] = code

            # Индексируем в master ветку
            indexer.index_file(
                file_path=file_path,
                repo_id="CoreDomain",
                project_id="orbita",
                branch="master",
                rel_path=f"src/{file_name}"
            )

        # 2. Создаем 3 ворктри задач, в каждой изменяем ровно по 2 файла
        task_modifications = {
            "orbit-1": [1, 2],    # service_01.py и service_02.py изменены
            "orbit-2": [3, 4],    # service_03.py и service_04.py изменены
            "orbit-3": [5, 6],    # service_05.py и service_06.py изменены
        }

        for task_name, modified_indices in task_modifications.items():
            task_dir = root / "worktrees" / task_name
            task_dir.mkdir(parents=True)

            for i in range(1, 51):
                file_name = f"service_{i:02d}.py"
                file_path = task_dir / file_name

                if i in modified_indices:
                    # Измененный контент для конкретной задачи
                    task_code = (
                        f"# Modified service #{i} specifically for task {task_name}\n"
                        f"def execute_domain_operation_{i:02d}(order_id: int) -> str:\n"
                        f"    \"\"\"Кастомная логика задачи {task_name}.\"\"\"\n"
                        f"    return f\"{task_name}_result_{{order_id}}\"\n"
                    )
                    file_path.write_text(task_code, encoding="utf-8")
                else:
                    # Идентичный контент из master (дедуплицируется в CAS!)
                    file_path.write_text(base_files_map[file_name], encoding="utf-8")

                # Индексируем в соответствующую ветку задачи
                indexer.index_file(
                    file_path=file_path,
                    repo_id="CoreDomain",
                    project_id="orbita",
                    branch=task_name,
                    rel_path=f"src/{file_name}"
                )

        # 3. Получаем статистику CAS-сжатия
        stats = db.get_cas_stats(project_id="orbita")

        # 4. Проверяем строгие математические инварианты дедупликации
        # Логических файлов ровно 200 (50 master + 50 * 3 задачи)
        assert stats["logical_files_count"] == 200, f"Ожидалось 200 логических файлов, получено {stats['logical_files_count']}"

        # Уникальных контентов в file_contents ровно 56:
        # 50 базовых файлов + 2 (orbit-1) + 2 (orbit-2) + 2 (orbit-3)
        assert stats["unique_contents_count"] == 56, f"Ожидалось 56 уникальных контентов, получено {stats['unique_contents_count']}"

        # Процент экономии: (200 - 56) / 200 = 144 / 200 = 72%
        assert stats["saved_percent"] >= 70.0, f"Сэкономленный процент должен превышать 70%, получено {stats['saved_percent']}%"
        assert stats["virtual_size_bytes"] > stats["real_size_bytes"] * 3, "Виртуальный объем должен более чем в 3 раза превышать физический"

        # 5. Проверяем таблицы chunks и symbols в SQLite:
        # В отличие от code-review-graph, где чанки и символы дублировались под каждый ворктри (4x = 200 записей),
        # благодаря content_hash в CAS чанки и символы создаются только для 56 уникальных файлов!
        cursor = db.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunks_count = cursor.fetchone()[0]
        assert chunks_count == 56, f"Чанки должны создаваться только для уникальных файлов (56), а не раздуваться до 200! Получено: {chunks_count}"

        cursor.execute("SELECT COUNT(*) FROM symbols")
        symbols_count = cursor.fetchone()[0]
        assert symbols_count == 56, f"Символы должны дедуплицироваться по content_hash (56), получено: {symbols_count}"

    def test_e2e_jit_freshness_guarantee(self, clean_env, monkeypatch):
        """
        Тест 3: Надежность JIT-синхронизации (JIT Freshness Guarantee).
        - Создается репозиторий, индексируется первоначальное состояние.
        - Файл редактируется разработчиком напрямую на диске (имитируя правку в Visual Studio / Rider).
        - Без коммита в git и без перезапуска сервиса вызываются MCP knowledge_search и knowledge_find_symbol.
        - JITSyncEngine обнаруживает несинхронизированное состояние (git status 2-3 мс),
          на лету переиндексирует файл, и инструменты MCP отдают актуальный свежий код.
        """
        root = clean_env["root"]
        db = clean_env["db"]

        repo_dir = root / "billing_service"
        repo_dir.mkdir(parents=True)

        subprocess.run(["git", "init", "-b", "master"], cwd=str(repo_dir), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Billing Dev"], cwd=str(repo_dir), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "dev@billing.com"], cwd=str(repo_dir), check=True, capture_output=True)

        service_file = repo_dir / "payment_gateway.py"
        service_file.write_text(
            "def process_standard_payment(order_id: int, amount: float) -> str:\n"
            "    \"\"\"Первоначальная версия шлюза.\"\"\"\n"
            "    return f\"paid_v1_legacy_{order_id}\"\n",
            encoding="utf-8"
        )

        subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit of billing"], cwd=str(repo_dir), check=True, capture_output=True)

        # Настраиваем конфигурацию
        config_data = {
            "repositories": [
                {
                    "id": "billing-repo",
                    "local_path": str(repo_dir),
                    "branch": "master",
                    "language": "python"
                }
            ],
            "projects": []
        }
        config_file = root / "config.yaml"
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")
        monkeypatch.setenv("CONFIG_PATH", str(config_file))

        # Первичная индексация
        indexer = Indexer(db, use_embeddings=False)
        indexer.sync_repo("billing-repo", repo_dir)

        # Проверяем начальный поиск
        s1 = knowledge_search("paid_v1_legacy", repo_ids=["billing-repo"], db=db)
        assert s1["total_results"] >= 1
        assert "paid_v1_legacy" in s1["results"][0]["content"]

        # Поиск несуществующей новой версии до правки возвращает 0
        s_pre = knowledge_search("paid_v2_instant_clearing", repo_ids=["billing-repo"], db=db)
        assert s_pre["total_results"] == 0

        # Разработчик вносит изменения в файл напрямую на диске
        updated_code = (
            "def process_standard_payment(order_id: int, amount: float) -> str:\n"
            "    \"\"\"Обновленная версия платежного шлюза с мгновенным клирингом.\"\"\"\n"
            "    return f\"paid_v2_instant_clearing_{order_id}\"\n\n"
            "def validate_crypto_wallet(wallet_address: str) -> bool:\n"
            "    \"\"\"Новая функция валидации кошелька.\"\"\"\n"
            "    return len(wallet_address) == 42\n"
        )
        service_file.write_text(updated_code, encoding="utf-8")

        # Моментальный вызов MCP knowledge_search БЕЗ ручной переиндексации
        s2 = knowledge_search("paid_v2_instant_clearing", repo_ids=["billing-repo"], db=db)
        assert s2["total_results"] >= 1, "JIT должен был автоматически обнаружить изменение и переиндексировать файл"
        assert "paid_v2_instant_clearing" in s2["results"][0]["content"]

        # Проверяем, что новый символ мгновенно доступен через knowledge_find_symbol
        sym_res = knowledge_find_symbol("validate_crypto_wallet", repo_ids=["billing-repo"], db=db)
        assert sym_res["total_symbols"] >= 1
        assert sym_res["symbols"][0]["name"] == "validate_crypto_wallet"
        assert sym_res["symbols"][0]["path"] == "payment_gateway.py"

        # Чтение файла через knowledge_read_file отдает свежий код
        fresh_content = knowledge_read_file("payment_gateway.py", repo_id="billing-repo", db=db)
        assert "paid_v2_instant_clearing" in fresh_content
        assert "validate_crypto_wallet" in fresh_content

    def test_e2e_code_review_blast_radius_and_test_gaps(self, clean_env):
        """
        Тест 4: Сквозной сценарий Code Review (Blast Radius, Test Gaps, Risk Scoring).
        - Репозиторий с OrderService (ProcessOrder, вызываемый ApiController),
          тестом OrderServiceTests (проверяет ProcessOrder) и новой непротестированной функцией FastCheckout.
        - Вызов knowledge_review_context для измененного файла Services/OrderService.cs.
        - Проверка результатов ревью:
          * Измененные файлы: Services/OrderService.cs.
          * Blast radius: ApiController затронут через вызов ProcessOrder.
          * Связанный тест OrderServiceTests найден через связь TESTED_BY.
          * Test Gap: FastCheckout отмечен как непротестированный метод без теста.
          * Скоринг риска: risk_score > 0, risk 'medium' или 'high'.
          * Метрика экономии токенов (context_savings) присутствует.
        """
        root = clean_env["root"]
        db = clean_env["db"]
        indexer = Indexer(db, use_embeddings=False)

        # 1. Создаем файлы C# сервиса, контроллера и теста
        svc_dir = root / "Services"
        ctrl_dir = root / "Controllers"
        test_dir = root / "Tests"
        svc_dir.mkdir(parents=True)
        ctrl_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)

        order_service_code = """
namespace Store.Services
{
    public class OrderService
    {
        public Order ProcessOrder(int orderId)
        {
            return new Order(orderId);
        }

        public Order FastCheckout(int orderId)
        {
            return new Order(orderId);
        }
    }
}
"""
        api_controller_code = """
namespace Store.Controllers
{
    public class ApiController
    {
        private readonly OrderService _service;
        public IActionResult PostOrder(int id)
        {
            return _service.ProcessOrder(id);
        }
    }
}
"""
        order_tests_code = """
using Xunit;

namespace Store.Tests
{
    public class OrderServiceTests
    {
        [Fact]
        public void Test_ProcessOrder()
        {
            var svc = new OrderService();
            svc.ProcessOrder(42);
        }
    }
}
"""
        svc_path = svc_dir / "OrderService.cs"
        ctrl_path = ctrl_dir / "ApiController.cs"
        test_path = test_dir / "OrderServiceTests.cs"

        svc_path.write_text(order_service_code, encoding="utf-8")
        ctrl_path.write_text(api_controller_code, encoding="utf-8")
        test_path.write_text(order_tests_code, encoding="utf-8")

        # 2. Индексируем файлы
        indexer.index_file(svc_path, repo_id="StoreCore", project_id="default", branch="master", rel_path="Services/OrderService.cs")
        indexer.index_file(ctrl_path, repo_id="StoreCore", project_id="default", branch="master", rel_path="Controllers/ApiController.cs")
        indexer.index_file(test_path, repo_id="StoreCore", project_id="default", branch="master", rel_path="Tests/OrderServiceTests.cs")

        # 3. Вызываем knowledge_review_context для анализа изменений в OrderService.cs
        review_res = knowledge_review_context(
            repo_id="StoreCore",
            project_id="default",
            branch="master",
            changed_files=["Services/OrderService.cs"],
            max_depth=2,
            db=db
        )

        # 4. Проверяем измененные файлы и символы
        assert review_res["changed_files"] == ["Services/OrderService.cs"]
        changed_sym_names = {s["name"] for s in review_res["changed_symbols"]}
        assert "ProcessOrder" in changed_sym_names
        assert "FastCheckout" in changed_sym_names

        # 5. Проверяем Blast Radius: ApiController затронут через вызов ProcessOrder
        blast = review_res["blast_radius"]
        assert blast["impacted_symbols_count"] >= 1
        impacted_files = {f.replace("\\", "/") for f in blast["impacted_files"]}
        assert "Controllers/ApiController.cs" in impacted_files
        caller_names = {c["name"] for c in blast["callers"]}
        assert any("PostOrder" in name for name in caller_names)

        # 6. Проверяем связанные тесты: OrderServiceTests найден
        rel_tests = review_res["related_tests"]
        assert len(rel_tests) >= 1
        rel_test_names = {t["test_name"] for t in rel_tests}
        assert any("OrderServiceTests" in t or "Test_ProcessOrder" in t for t in rel_test_names)

        # 7. Проверяем выявление Test Gap:
        # FastCheckout НЕ протестирован, тогда как ProcessOrder покрыт тестом
        test_gaps = review_res["test_gaps"]
        test_gap_names = {g["name"] for g in test_gaps}
        assert "FastCheckout" in test_gap_names, "FastCheckout должен быть помечен как Test Gap"
        assert "ProcessOrder" not in test_gap_names, "ProcessOrder покрыт тестом и не должен быть Test Gap"

        # 8. Проверяем скоринг риска
        assert review_res["risk_score"] > 0
        assert review_res["risk"] in ("medium", "high")

        # 9. Проверяем расчет сэкономленных токенов
        savings = review_res["context_savings"]
        assert savings["saved_tokens"] > 0
        assert savings["saved_percent"] > 0
        assert savings["baseline_tokens"] > savings["returned_tokens"]

        # 10. Проверяем вызов через MCP протокол call_tool
        mcp_res = asyncio.run(call_tool("knowledge_review_context", {
            "repo_id": "StoreCore",
            "project_id": "default",
            "branch": "master",
            "changed_files": ["Services/OrderService.cs"]
        }))
        assert len(mcp_res) == 1
        payload = json.loads(mcp_res[0].text)
        assert payload["repo_id"] == "StoreCore"
        assert len(payload["test_gaps"]) >= 1
        assert payload["risk"] in ("medium", "high")
