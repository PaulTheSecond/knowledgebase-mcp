import anyio
import logging
from mcp.server.stdio import stdio_server
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)

async def run_stdio_sse_proxy(sse_url: str):
    """Проксирует MCP сообщения между локальным STDIO процессом и удаленным SSE эндпоинтом.
    
    Предотвращает множественный доступ к SQLite базе данных на хосте,
    маршрутизируя все запросы MCP клиентов (Claude Code, Antigravity) в единый демон-писатель.
    """
    async with stdio_server() as (stdio_read, stdio_write):
        async with sse_client(sse_url, timeout=60.0, sse_read_timeout=1800.0) as (sse_read, sse_write):
            async with anyio.create_task_group() as tg:
                async def forward_stdio_to_sse():
                    try:
                        async for msg in stdio_read:
                            if isinstance(msg, Exception):
                                raise msg
                            await sse_write.send(msg)
                    finally:
                        tg.cancel_scope.cancel()

                async def forward_sse_to_stdio():
                    try:
                        async for msg in sse_read:
                            if isinstance(msg, Exception):
                                raise msg
                            await stdio_write.send(msg)
                    finally:
                        tg.cancel_scope.cancel()

                tg.start_soon(forward_stdio_to_sse)
                tg.start_soon(forward_sse_to_stdio)
