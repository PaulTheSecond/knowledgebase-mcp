import os
import logging
import httpx
import json
from typing import List
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

    api_key = os.environ.get("LLM_API_KEY")
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
    sources = [{"filepath": c["path"], "lines": f"{c.get('line_start')}-{c.get('line_end')}"} for c in chunks]

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
                    "messages": formatted_messages
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
                
            res_data = response.json()
            content = res_data["choices"][0]["message"]["content"]
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

    api_key = os.environ.get("LLM_API_KEY")
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
    sources = [{"filepath": c["path"], "lines": f"{c.get('line_start')}-{c.get('line_end')}"} for c in chunks]

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


@router.get("/file")
async def read_file_endpoint(
    filepath: str, 
    start_line: int = 1, 
    end_line: int = None, 
    username: str = Depends(verify_basic_auth)
):
    """
    Возвращает содержимое указанного файла (или фрагмент строк).
    Защищает от Path Traversal, проверяя допустимые директории и блокируя доступ к конфигам и секретам.
    """
    from pathlib import Path
    
    # 1. Защита от Path Traversal и спецсимволов
    if ".." in filepath.replace("\\", "/").split("/"):
        raise HTTPException(status_code=403, detail="Access denied: Directory traversal sequence detected")

    clones_dir = Path(os.environ.get("CLONES_DIR", "data/clones")).resolve()
    project_dir = Path(".").resolve()
    
    # 2. Приводим путь к абсолютному виду
    target_path = (clones_dir / filepath).resolve()
    if not target_path.exists():
        target_path = (project_dir / filepath).resolve()
        
    # 3. Проверка безопасности: путь должен начинаться либо с clones_dir, либо с project_dir
    try:
        is_inside_clones = target_path.is_relative_to(clones_dir)
        is_inside_project = target_path.is_relative_to(project_dir)
    except AttributeError:
        # Для обратной совместимости с Python < 3.9 (хотя у нас 3.14)
        is_inside_clones = target_path.parts[:len(clones_dir.parts)] == clones_dir.parts
        is_inside_project = target_path.parts[:len(project_dir.parts)] == project_dir.parts

    if not (is_inside_clones or is_inside_project):
        raise HTTPException(status_code=403, detail="Access denied: File is outside allowed directories")

    # 4. Блокировка доступа к секретам, конфигурационным файлам и системным базам данных
    target_name = target_path.name.lower()
    target_suffix = target_path.suffix.lower()
    parts_lower = [p.lower() for p in target_path.parts]

    if (
        target_name in SENSITIVE_FILES
        or target_suffix in SENSITIVE_EXTENSIONS
        or any(part in (".git", ".ssh", ".idea", ".vscode") for part in parts_lower)
        or (is_inside_project and not is_inside_clones and "data" in parts_lower and target_suffix in (".db", ".yaml", ".yml"))
    ):
        raise HTTPException(status_code=403, detail="Access denied: Access to sensitive or configuration files is prohibited")
        
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
        
    # 5. Читаем строки
    try:
        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            
        total_lines = len(lines)
        start = max(1, start_line) - 1
        end = min(total_lines, end_line) if end_line else total_lines
        
        selected_lines = lines[start:end]
        return {
            "content": "".join(selected_lines),
            "filepath": filepath,
            "start_line": start + 1,
            "end_line": end,
            "total_lines": total_lines
        }
    except Exception as e:
        logger.exception("Failed to read file")
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")
