import os
import logging
import httpx
import json
from pathlib import Path
from typing import List, Optional, Tuple
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from knowledge_mcp.daemon.auth import verify_basic_auth
from knowledge_mcp.core.config import load_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

def verify_query_token(token: str = None):
    from knowledge_mcp.daemon.auth import verify_temp_token
    if not token or not verify_temp_token(token):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return token

def get_components():
    """Получает актуальные инстансы (db, indexer) из main_mod или __main__."""
    import sys
    import knowledge_mcp.main as main_mod
    current_db = getattr(main_mod, "db", None) or getattr(sys.modules.get("__main__"), "db", None)
    current_indexer = getattr(main_mod, "indexer", None) or getattr(sys.modules.get("__main__"), "indexer", None)
    return current_db, current_indexer


def get_llm_settings(config: dict) -> tuple[str, str]:
    """Возвращает (endpoint_url, model_name) для обращения к LLM/OmniRoute.
    
    Приоритет:
    1. Переменные окружения LLM_BASE_URL и LLM_MODEL.
    2. Секция server.dashboard.chat в config.yaml (api_base, model).
    3. Дефолты OmniRoute (http://localhost:20128/v1/chat/completions, kr/claude-sonnet-4.5).
    """
    chat_cfg = config.get("server", {}).get("dashboard", {}).get("chat", {})
    
    base_url = os.environ.get("LLM_BASE_URL") or chat_cfg.get("api_base") or "http://localhost:20128/v1"
    if os.path.exists("/.dockerenv"):
        base_url = (
            base_url
            .replace("://127.0.0.1:", "://host.docker.internal:")
            .replace("://localhost:", "://host.docker.internal:")
        )
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/chat/completions"):
        endpoint_url = f"{base_url}/chat/completions"
    else:
        endpoint_url = base_url

    model = os.environ.get("LLM_MODEL") or chat_cfg.get("model") or "kr/claude-sonnet-4.5"
    return endpoint_url, model


@router.post("")
async def sync_chat_endpoint(req: ChatRequest, username: str = Depends(verify_basic_auth)):
    """Синхронный RAG Чат. Проводит гибридный поиск и запрашивает OmniRoute/LLM."""
    config = load_config()
    chat_enabled = config.get("server", {}).get("dashboard", {}).get("chat", {}).get("enabled", False)
    if not chat_enabled:
        raise HTTPException(status_code=403, detail="Chat is disabled")

    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    user_query = user_messages[-1].content

    # Получаем зависимости через get_components
    db_inst, indexer_inst = get_components()
    if not db_inst or not indexer_inst:
        raise HTTPException(status_code=500, detail="Search engine not initialized")

    chat_cfg = config.get("server", {}).get("dashboard", {}).get("chat", {})
    api_key = os.environ.get("LLM_API_KEY") or chat_cfg.get("api_key")
    if not api_key:
        raise HTTPException(status_code=500, detail="LLM_API_KEY environment variable is not set")

    # 1. Вычисляем вектор вопроса (если инициализирован embedder)
    query_vector = None
    if indexer_inst.embedder:
        try:
            query_vector = indexer_inst.embedder.embed_text(user_query, is_query=True)
        except Exception as e:
            logger.warning(f"Failed to calculate query vector: {e}")

    # 2. Выполняем гибридный поиск по базе знаний (FTS5 + Graph, плюс векторы если доступны)
    try:
        raw_chunks = db_inst.search_chunks_hybrid(user_query, query_vector, limit=5)
        chunks = [dict(c) if not isinstance(c, dict) else c for c in raw_chunks]
    except Exception:
        logger.exception("Hybrid search failed")
        chunks = []

    # 3. Сборка контекста для системного промпта
    context_str = ""
    for chunk in chunks:
        path = chunk["path"]
        line_start = chunk.get("line_start", 1)
        line_end = chunk.get("line_end", 1)
        content = chunk["content"]
        context_str += f"\n[FILEPATH: {path}]\n[LINES: {line_start}-{line_end}]\n{content}\n"

    system_prompt = f"""Используй следующий контекст исходного кода для ответа на вопрос.
Если в контексте нет информации для ответа, ответь на основе своих знаний, но упомяни, что в кодовой базе релевантный контекст не найден.

Контекст:
{context_str}
"""

    # 4. OpenAI-совместимый вызов LLM через HTTP-запросы
    formatted_messages = [{"role": "system", "content": system_prompt}]
    for m in req.messages:
        formatted_messages.append({"role": m.role, "content": m.content})

    endpoint_url, model_name = get_llm_settings(config)
    sources = [
        {
            "repo_id": c.get("repo_id"),
            "filepath": c["path"],
            "lines": f"{c.get('line_start')}-{c.get('line_end')}",
            "line_start": c.get("line_start"),
            "line_end": c.get("line_end"),
        }
        for c in chunks
    ]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                endpoint_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model_name,
                    "messages": formatted_messages,
                    "stream": False
                },
                timeout=60.0
            )
            
            if response.status_code != 200:
                logger.error(f"LLM HTTP error {response.status_code}: {response.text}")
                return {
                    "message": {
                        "role": "assistant",
                        "content": f"⚠️ **Ошибка ответа LLM (HTTP {response.status_code})**\n\n"
                                   f"**Эндпоинт:** `{endpoint_url}`\n**Модель:** `{model_name}`\n\n"
                                   f"**Детали:** {response.text}\n\n"
                                   f"**Найденные фрагменты в базе знаний ({len(sources)} шт.):**\n"
                                   + ("\n".join([f"- `{s['filepath']}:{s['lines']}`" for s in sources]) if sources else "Релевантный код в индексе не найден.")
                    },
                    "sources": sources
                }
                
            try:
                res_data = response.json()
                content = res_data["choices"][0]["message"]["content"]
            except Exception:
                # Если LLM-прокси вернул SSE-чанки в синхронном ответе
                lines = response.text.splitlines()
                collected = []
                for line in lines:
                    line = line.strip()
                    if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                        try:
                            cj = json.loads(line[6:])
                            if "choices" in cj and len(cj["choices"]) > 0:
                                delta = cj["choices"][0].get("delta", {}) or cj["choices"][0].get("message", {})
                                c = delta.get("content", "")
                                if c:
                                    collected.append(c)
                        except Exception:
                            pass
                content = "".join(collected) if collected else response.text

            return {
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "sources": sources
            }
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError) as e:
        logger.warning(f"Cannot connect to LLM at {endpoint_url}: {e}")
        return {
            "message": {
                "role": "assistant",
                "content": f"⚠️ **Не удалось подключиться к LLM-шлюзу** по адресу `{endpoint_url}`.\n\n"
                           f"**Модель:** `{model_name}`\n\n"
                           f"Убедитесь, что сервер инференса (например, OmniRoute, vLLM, Ollama или LocalAI) запущен и доступен по указанному адресу.\n\n"
                           f"**Найденные релевантные фрагменты в базе знаний ({len(sources)} шт.):**\n"
                           + ("\n".join([f"- `{s['filepath']}:{s['lines']}`" for s in sources]) if sources else "Релевантный код в индексе не найден.")
            },
            "sources": sources
        }
    except Exception as e:
        logger.exception("LLM request failed")
        if isinstance(e, HTTPException):
            raise
        return {
            "message": {
                "role": "assistant",
                "content": f"⚠️ Ошибка вызова LLM: {str(e)}"
            },
            "sources": sources
        }


@router.post("/stream")
async def stream_chat_endpoint(req: ChatRequest, token: str = Depends(verify_query_token)):
    """SSE Стриминг RAG Чат. Возвращает StreamingResponse."""
    config = load_config()
    chat_enabled = config.get("server", {}).get("dashboard", {}).get("chat", {}).get("enabled", False)
    if not chat_enabled:
        raise HTTPException(status_code=403, detail="Chat is disabled")

    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    user_query = user_messages[-1].content

    # Получаем зависимости через get_components
    db_inst, indexer_inst = get_components()
    if not db_inst or not indexer_inst:
        raise HTTPException(status_code=500, detail="Search engine not initialized")

    chat_cfg = config.get("server", {}).get("dashboard", {}).get("chat", {})
    api_key = os.environ.get("LLM_API_KEY") or chat_cfg.get("api_key")
    if not api_key:
        raise HTTPException(status_code=500, detail="LLM_API_KEY environment variable is not set")

    # 1. Вычисляем вектор вопроса (если инициализирован embedder)
    query_vector = None
    if indexer_inst.embedder:
        try:
            query_vector = indexer_inst.embedder.embed_text(user_query, is_query=True)
        except Exception as e:
            logger.warning(f"Failed to calculate query vector: {e}")

    # 2. Выполняем гибридный поиск по базе знаний (FTS5 + Graph, плюс векторы если есть)
    try:
        raw_chunks = db_inst.search_chunks_hybrid(user_query, query_vector, limit=5)
        chunks = [dict(c) if not isinstance(c, dict) else c for c in raw_chunks]
    except Exception:
        logger.exception("Hybrid search failed")
        chunks = []

    # 3. Сборка контекста для системного промпта
    context_str = ""
    for chunk in chunks:
        path = chunk["path"]
        line_start = chunk.get("line_start", 1)
        line_end = chunk.get("line_end", 1)
        content = chunk["content"]
        context_str += f"\n[FILEPATH: {path}]\n[LINES: {line_start}-{line_end}]\n{content}\n"

    system_prompt = f"""Используй следующий контекст исходного кода для ответа на вопрос.
Если в контексте нет информации для ответа, ответь на основе своих знаний, но упомяни, что в кодовой базе релевантный контекст не найден.

Контекст:
{context_str}
"""

    formatted_messages = [{"role": "system", "content": system_prompt}]
    for m in req.messages:
        formatted_messages.append({"role": m.role, "content": m.content})

    endpoint_url, model_name = get_llm_settings(config)
    sources = [
        {
            "repo_id": c.get("repo_id"),
            "filepath": c["path"],
            "lines": f"{c.get('line_start')}-{c.get('line_end')}",
            "line_start": c.get("line_start"),
            "line_end": c.get("line_end"),
        }
        for c in chunks
    ]

    async def event_generator():
        try:
            async with httpx.AsyncClient() as client:
                try:
                    async with client.stream(
                        "POST",
                        endpoint_url,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": model_name,
                            "messages": formatted_messages,
                            "stream": True
                        },
                        timeout=60.0
                    ) as response:
                        if response.status_code != 200:
                            err_text = await response.aread()
                            err_msg = (
                                f"⚠️ **Ошибка LLM-сервера (HTTP {response.status_code})**\n\n"
                                f"**Эндпоинт:** `{endpoint_url}`\n**Модель:** `{model_name}`\n\n"
                                f"**Детали:** {err_text.decode('utf-8', errors='replace')}\n\n"
                                f"**Найденные фрагменты кода в базе знаний:**"
                            )
                            yield f"data: {json.dumps({'token': err_msg})}\n\n"
                            yield f"data: {json.dumps({'sources': sources})}\n\n"
                            return

                        async for line in response.aiter_lines():
                            if not line.strip():
                                continue
                            if line.startswith("data: "):
                                data_content = line[6:]
                                if data_content.strip() == "[DONE]":
                                    break
                                try:
                                    chunk_json = json.loads(data_content)
                                    if "choices" in chunk_json and len(chunk_json["choices"]) > 0:
                                        delta = chunk_json["choices"][0].get("delta", {})
                                        delta_content = delta.get("content")
                                        if delta_content:
                                            yield f"data: {json.dumps({'token': delta_content})}\n\n"
                                except Exception as e:
                                    logger.warning(f"Error parsing SSE chunk: {e}")

                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError) as e:
                    logger.warning(f"LLM connection error at {endpoint_url}: {e}")
                    conn_msg = (
                        f"⚠️ **Не удалось подключиться к LLM-шлюзу** по адресу `{endpoint_url}`.\n\n"
                        f"**Модель:** `{model_name}`\n\n"
                        f"Убедитесь, что сервер инференса (например, OmniRoute, vLLM, Ollama или LocalAI) запущен и доступен по указанному адресу.\n\n"
                        f"**Найденные релевантные фрагменты в базе знаний ({len(sources)} шт.):**"
                    )
                    yield f"data: {json.dumps({'token': conn_msg})}\n\n"

            # Отправка источников в конце
            yield f"data: {json.dumps({'sources': sources})}\n\n"

        except Exception as e:
            logger.exception("Error in event generator")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Список защищенных файлов и расширений, доступ к которым категорически запрещен через UI
SENSITIVE_FILES = {
    ".env", ".env.local", ".env.production",
    "config.yaml", "config.user.toml", "settings.json",
    "knowledge.db", "knowledge.db-wal", "knowledge.db-shm"
}
SENSITIVE_EXTENSIONS = {".db", ".sqlite", ".sqlite3", ".key", ".pem", ".pfx", ".token"}


def get_configured_repo_roots(config: dict) -> List[Tuple[Optional[str], Path]]:
    """Возвращает список корней репозиториев (repo_id, absolute_path) для одиночных репо и проектных воркспейсов."""
    repos = config.get("repositories", [])
    projects = config.get("projects", [])
    clones_dir = Path(os.environ.get("CLONES_DIR", "data/clones")).resolve()
    repos_dir_env = os.environ.get("REPOS_DIR")

    roots: List[Tuple[Optional[str], Path]] = []

    # 1. Одиночные настроенные репозитории
    for r in repos:
        repo_id = r.get("id")
        local_path_str = r.get("local_path")
        resolved_path = None

        if local_path_str:
            p = Path(local_path_str)
            if p.exists():
                resolved_path = p.resolve()
            elif repos_dir_env:
                clean_name = local_path_str.replace("\\", "/").rstrip("/").split("/")[-1]
                candidate = Path(repos_dir_env) / clean_name
                if candidate.exists():
                    resolved_path = candidate.resolve()
            if not resolved_path:
                clean_name = local_path_str.replace("\\", "/").rstrip("/").split("/")[-1]
                for prefix in (Path("C:/repos"), Path("/repos")):
                    candidate = prefix / clean_name
                    if candidate.exists():
                        resolved_path = candidate.resolve()
                        break

        if not resolved_path and repo_id:
            clone_path = clones_dir / repo_id
            if clone_path.exists():
                resolved_path = clone_path.resolve()

        if resolved_path:
            roots.append((repo_id, resolved_path))

    # 2. Проектные воркспейсы (projects) с поддержкой worktree и docs_paths
    for proj in projects:
        proj_id = proj.get("id")
        ws_root_val = proj.get("workspace_root")
        if not ws_root_val:
            continue

        ws_root = None
        p = Path(ws_root_val)
        if p.exists():
            ws_root = p.resolve()
        else:
            clean_ws = ws_root_val.replace("\\", "/")
            if "repos/" in clean_ws.lower():
                rel_tail = clean_ws.lower().split("repos/", 1)[-1].lstrip("/")
                candidates = []
                if repos_dir_env:
                    candidates.append(Path(repos_dir_env) / rel_tail)
                candidates.extend([Path("C:/repos") / rel_tail, Path("c:/repos") / rel_tail, Path("/repos") / rel_tail])
                for cand in candidates:
                    if cand.exists():
                        ws_root = cand.resolve()
                        break

        if not ws_root or not ws_root.exists():
            continue

        roots.append((proj_id, ws_root))

        base_rel = proj.get("base_path", "worktrees/master")
        base_dir = (ws_root / base_rel).resolve()
        tasks_parent = (ws_root / "worktrees").resolve()

        for repo in proj.get("repositories", []):
            r_id = repo.get("id") or repo.get("path")
            r_rel = repo.get("path", "")

            # 1) Базовое дерево (например, worktrees/master/backend)
            if base_dir.exists():
                cand = (base_dir / r_rel).resolve()
                if cand.exists() and (r_id, cand) not in roots:
                    roots.append((r_id, cand))

            # 2) Напрямую в корне воркспейса
            cand_ws = (ws_root / r_rel).resolve()
            if cand_ws.exists() and (r_id, cand_ws) not in roots:
                roots.append((r_id, cand_ws))

            # 3) Другие ворктри задач (worktrees/*)
            if tasks_parent.exists() and tasks_parent.is_dir():
                for t_dir in tasks_parent.iterdir():
                    if t_dir.is_dir() and t_dir != base_dir:
                        cand_task = (t_dir / r_rel).resolve()
                        if cand_task.exists() and (r_id, cand_task) not in roots:
                            roots.append((r_id, cand_task))

        # Документация проекта (.kb, CLAUDE.md и т.д.)
        for doc_rel in proj.get("docs_paths", []):
            doc_cand = (ws_root / doc_rel).resolve()
            if doc_cand.exists():
                if doc_cand.is_dir():
                    if ("docs", doc_cand) not in roots:
                        roots.append(("docs", doc_cand))
                    if (proj_id, doc_cand) not in roots:
                        roots.append((proj_id, doc_cand))
                else:
                    if ("docs", doc_cand) not in roots:
                        roots.append(("docs", doc_cand))
                    if (proj_id, doc_cand) not in roots:
                        roots.append((proj_id, doc_cand))

    return roots


def _is_path_safe(target_path: Path, allowed_roots: List[Path]) -> bool:
    """Проверяет, что путь находится внутри одной из разрешенных корневых директорий."""
    for root in allowed_roots:
        try:
            if target_path.is_relative_to(root):
                return True
        except AttributeError:
            if target_path.parts[:len(root.parts)] == root.parts:
                return True
    return False


@router.get("/file")
async def read_file_endpoint(
    filepath: str,
    repo_id: Optional[str] = None,
    start_line: int = 1,
    end_line: Optional[int] = None,
    full_file: bool = False,
    username: str = Depends(verify_basic_auth),
):
    """Возвращает содержимое указанного файла (весь файл или фрагмент строк).

    Поддерживает поиск по всем подключенным репозиториям и безопасно проверяет
    пути.
    """
    # 1. Защита от Path Traversal и спецсимволов
    if ".." in filepath.replace("\\", "/").split("/"):
        raise HTTPException(
            status_code=403,
            detail="Access denied: Directory traversal sequence detected",
        )

    config = load_config()
    repo_roots = get_configured_repo_roots(config)
    clones_dir = Path(os.environ.get("CLONES_DIR", "data/clones")).resolve()
    project_dir = Path(".").resolve()

    # Разрешенные корневые директории для проверки безопасности
    allowed_roots = [project_dir, clones_dir] + [root for _, root in repo_roots]

    target_path: Optional[Path] = None
    found_repo_id: Optional[str] = repo_id
    clean_filepath = filepath.replace("\\", "/").lstrip("/")

    # А. Если передан конкретный repo_id, ищем сначала в нём
    if repo_id:
        for r_id, r_root in repo_roots:
            if r_id == repo_id:
                if r_root.is_file():
                    if clean_filepath == r_root.name or clean_filepath.endswith("/" + r_root.name):
                        target_path = r_root
                        break
                    continue
                candidate = (r_root / clean_filepath).resolve()
                if candidate.is_file():
                    target_path = candidate
                    break
                # Пробуем снять префикс repo_id/ или r_root.name/
                for pfx in (f"{repo_id}/", f"{r_root.name}/"):
                    if clean_filepath.startswith(pfx):
                        candidate = (r_root / clean_filepath[len(pfx):]).resolve()
                        if candidate.is_file():
                            target_path = candidate
                            break
                if target_path:
                    break
        if not target_path and repo_id == "knowledgebase-mcp":
            candidate = (project_dir / clean_filepath).resolve()
            if candidate.is_file():
                target_path = candidate

    # Б. Если repo_id не указан или файл в целевом репо не найден:
    if not target_path:
        for r_id, r_root in repo_roots:
            if r_root.is_file():
                if clean_filepath == r_root.name or clean_filepath.endswith("/" + r_root.name):
                    target_path = r_root
                    found_repo_id = r_id
                    break
                continue
            candidate = (r_root / clean_filepath).resolve()
            if candidate.is_file():
                target_path = candidate
                found_repo_id = r_id
                break
            if r_id and clean_filepath.startswith(f"{r_id}/"):
                candidate = (r_root / clean_filepath[len(r_id) + 1:]).resolve()
                if candidate.is_file():
                    target_path = candidate
                    found_repo_id = r_id
                    break

    # В. Проверяем clones_dir
    if not target_path:
        candidate = (clones_dir / clean_filepath).resolve()
        if candidate.is_file():
            target_path = candidate

    # Г. Проверяем project_dir
    if not target_path:
        candidate = (project_dir / clean_filepath).resolve()
        if candidate.is_file():
            target_path = candidate

    # Д. Поиск в подпапках корней репозиториев с отсечением служебных каталогов
    if not target_path:
        filename = Path(clean_filepath).name
        candidate_roots = [
            (r_id, r_root) for r_id, r_root in repo_roots if r_id == repo_id
        ] if repo_id else repo_roots

        exclude_dirs = {
            ".git", "node_modules", "bin", "obj", ".idea", ".vscode",
            "dist", "build", ".venv", "venv", "__pycache__", ".cache"
        }

        for r_id, r_root in candidate_roots:
            if not r_root.is_dir():
                continue
            try:
                for cur_dir, dirs, files in os.walk(r_root):
                    # Отсекаем служебные и скрытые папки
                    dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]
                    if filename in files:
                        cand = (Path(cur_dir) / filename).resolve()
                        if cand.is_file() and _is_path_safe(cand, allowed_roots):
                            target_path = cand
                            found_repo_id = r_id
                            break
                if target_path:
                    break
            except Exception:
                continue

    # Е. Fallback на чтение из CAS-хранилища базы данных (таблица file_contents)
    db_content: Optional[str] = None
    if not target_path:
        try:
            filename = Path(clean_filepath).name
            db_path = os.environ.get("DB_PATH", os.environ.get("KNOWLEDGE_DB_PATH", config.get("db_path", "/data/knowledge.db")))
            if not Path(db_path).exists():
                db_path = "data/knowledge.db" if Path("data/knowledge.db").exists() else "knowledge.db"

            if Path(db_path).exists():
                import sqlite3
                sconn = sqlite3.connect(db_path)
                try:
                    sc = sconn.cursor()
                    q = """
                        SELECT fc.content, f.repo_id
                        FROM files f
                        JOIN file_contents fc ON (fc.hash = f.content_hash OR fc.hash = f.hash)
                        WHERE (f.path = ? OR f.path LIKE ? OR f.path LIKE ?)
                    """
                    p = [clean_filepath, f"%/{filename}", f"%\\{filename}"]
                    if repo_id:
                        q_with_repo = q + " AND (f.repo_id = ? OR f.repo_id = 'docs') AND LENGTH(fc.content) > 0 LIMIT 1"
                        sc.execute(q_with_repo, p + [repo_id])
                        r = sc.fetchone()
                    else:
                        r = None

                    if not r or not r[0]:
                        q_any = q + " AND LENGTH(fc.content) > 0 LIMIT 1"
                        sc.execute(q_any, p)
                        r = sc.fetchone()

                    if r and r[0]:
                        db_content = r[0]
                        found_repo_id = r[1] or found_repo_id
                finally:
                    sconn.close()
        except Exception as e:
            logger.debug(f"Database CAS fallback check failed: {e}")

    # 3. Если файл не найден ни на диске, ни в базе данных
    if not target_path and db_content is None:
        raise HTTPException(status_code=404, detail="File not found")

    # 4. Если файл получен из CAS базы данных
    if db_content is not None:
        lines = db_content.splitlines(keepends=True)
        total_lines = len(lines)
        start = max(1, start_line) - 1
        end = min(total_lines, end_line) if end_line else total_lines
        content = "".join(lines) if full_file else "".join(lines[start:end])
        return {
            "content": content,
            "filepath": filepath,
            "repo_id": found_repo_id,
            "start_line": start + 1,
            "end_line": end,
            "total_lines": total_lines,
            "is_full_file": full_file,
        }

    # 4. Проверка безопасности: путь должен начинаться с одного из разрешенных корней
    if not _is_path_safe(target_path, allowed_roots):
        raise HTTPException(
            status_code=403,
            detail="Access denied: File is outside allowed directories",
        )

    # 5. Блокировка доступа к секретам, конфигурационным файлам и системным базам данных
    target_name = target_path.name.lower()
    target_suffix = target_path.suffix.lower()
    parts_lower = [p.lower() for p in target_path.parts]

    if (
        target_name in SENSITIVE_FILES
        or target_suffix in SENSITIVE_EXTENSIONS
        or any(part in (".git", ".ssh", ".idea", ".vscode") for part in parts_lower)
        or ("data" in parts_lower and target_suffix in (".db", ".yaml", ".yml"))
    ):
        raise HTTPException(
            status_code=403,
            detail="Access denied: Access to sensitive or configuration files is prohibited",
        )

    # 6. Читаем файл
    try:
        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        start = max(1, start_line) - 1
        end = min(total_lines, end_line) if end_line else total_lines

        if full_file:
            content = "".join(lines)
        else:
            content = "".join(lines[start:end])

        return {
            "content": content,
            "filepath": filepath,
            "repo_id": found_repo_id,
            "start_line": start + 1,
            "end_line": end,
            "total_lines": total_lines,
            "is_full_file": full_file,
        }
    except Exception as e:
        logger.exception("Failed to read file")
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")
