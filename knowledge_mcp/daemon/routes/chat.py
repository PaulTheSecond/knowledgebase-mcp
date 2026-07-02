import os
import logging
import httpx
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
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
