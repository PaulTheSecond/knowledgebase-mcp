import os
import logging
import httpx
import json
from typing import List, Dict, Any
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

@router.post("")
async def sync_chat_endpoint(req: ChatRequest, username: str = Depends(verify_basic_auth)):
    """Синхронный RAG Чат. Проводит гибридный поиск и запрашивает OmniRoute."""
    config = load_config()
    chat_enabled = config.get("server", {}).get("dashboard", {}).get("chat", {}).get("enabled", False)
    if not chat_enabled:
        raise HTTPException(status_code=403, detail="Chat is disabled")

    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    user_query = user_messages[-1].content

    # Получаем зависимости из main.py
    import knowledge_mcp.main as main_mod
    if not main_mod.db or not main_mod.indexer or not main_mod.indexer.embedder:
        raise HTTPException(status_code=500, detail="Search engine or Embedder not initialized")

    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
         raise HTTPException(status_code=500, detail="LLM_API_KEY environment variable is not set")

    # 1. Вычисляем вектор вопроса
    query_vector = main_mod.indexer.embedder.embed_text(user_query, is_query=True)

    # 2. Выполняем гибридный поиск по базе знаний
    try:
        chunks = main_mod.db.search_chunks_hybrid(user_query, query_vector, limit=5)
    except Exception as e:
        logger.exception("Hybrid search failed")
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {e}")

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

    # 4. OpenAI-совместимый вызов OmniRoute через HTTP-запросы
    formatted_messages = [{"role": "system", "content": system_prompt}]
    for m in req.messages:
        formatted_messages.append({"role": m.role, "content": m.content})

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:20128/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "kr/claude-sonnet-4.5",
                    "messages": formatted_messages
                },
                timeout=60.0
            )
            
            if response.status_code != 200:
                logger.error(f"OmniRoute HTTP error {response.status_code}: {response.text}")
                raise HTTPException(status_code=500, detail=f"OmniRoute error: {response.text}")
                
            res_data = response.json()
            content = res_data["choices"][0]["message"]["content"]
        
        sources = [{"filepath": c["path"], "lines": f"{c.get('line_start')}-{c.get('line_end')}"} for c in chunks]
        return {
            "message": {
                "role": "assistant",
                "content": content
            },
            "sources": sources
        }
    except Exception as e:
        logger.exception("OmniRoute request failed")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"OmniRoute request failed: {e}")

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

    # Получаем зависимости из main.py
    import knowledge_mcp.main as main_mod
    if not main_mod.db or not main_mod.indexer or not main_mod.indexer.embedder:
        raise HTTPException(status_code=500, detail="Search engine or Embedder not initialized")

    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
         raise HTTPException(status_code=500, detail="LLM_API_KEY environment variable is not set")

    # 1. Вычисляем вектор вопроса
    query_vector = main_mod.indexer.embedder.embed_text(user_query, is_query=True)

    # 2. Выполняем гибридный поиск по базе знаний
    try:
        chunks = main_mod.db.search_chunks_hybrid(user_query, query_vector, limit=5)
    except Exception as e:
        logger.exception("Hybrid search failed")
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {e}")

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

    async def event_generator():
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    "http://localhost:20128/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "kr/claude-sonnet-4.5",
                        "messages": formatted_messages,
                        "stream": True
                    },
                    timeout=60.0
                ) as response:
                    if response.status_code != 200:
                        err_text = await response.aread()
                        yield f"data: {json.dumps({'error': f'OmniRoute stream error: {err_text.decode()}'})}\n\n"
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

            # Отправка источников в конце
            sources = [{"filepath": c["path"], "lines": f"{c.get('line_start')}-{c.get('line_end')}"} for c in chunks]
            yield f"data: {json.dumps({'sources': sources})}\n\n"

        except Exception as e:
            logger.exception("Error in event generator")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/file")
async def read_file_endpoint(
    filepath: str, 
    start_line: int = 1, 
    end_line: int = None, 
    username: str = Depends(verify_basic_auth)
):
    """
    Возвращает содержимое указанного файла (или фрагмент строк).
    Защищает от Path Traversal, проверяя, что путь находится внутри CLONES_DIR или папки проекта.
    """
    from pathlib import Path
    clones_dir = Path(os.environ.get("CLONES_DIR", "data/clones")).resolve()
    project_dir = Path(".").resolve()
    
    # 1. Приводим путь к абсолютному виду
    # Сначала пытаемся разрешить путь относительно clones_dir
    target_path = (clones_dir / filepath).resolve()
    
    # Если файл не существует относительно clones_dir, пробуем относительно корня проекта
    if not target_path.exists():
        target_path = (project_dir / filepath).resolve()
        
    # 2. Проверка безопасности (Path Traversal)
    # Путь должен начинаться либо с clones_dir, либо с project_dir
    try:
        is_inside_clones = target_path.parts[:len(clones_dir.parts)] == clones_dir.parts
        is_inside_project = target_path.parts[:len(project_dir.parts)] == project_dir.parts
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file path structure")
    
    if not (is_inside_clones or is_inside_project):
        raise HTTPException(status_code=403, detail="Access denied: File is outside allowed directories")
        
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
        
    # 3. Читаем строки
    try:
        with open(target_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            
        total_lines = len(lines)
        # Корректируем start_line и end_line (1-indexed)
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
