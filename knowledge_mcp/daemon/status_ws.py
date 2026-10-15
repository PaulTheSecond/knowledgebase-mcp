import asyncio
import logging
from typing import List
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        logger.debug(f"Broadcasting to WS: {message}")
        connections = list(self.active_connections)
        if not connections:
            return

        async def _send_safe(ws: WebSocket):
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")
                self.disconnect(ws)

        await asyncio.gather(*(_send_safe(c) for c in connections), return_exceptions=True)

manager = ConnectionManager()

def broadcast_sync(message: dict):
    """Синхронный мост для отправки сообщений в вебсокет из фоновых потоков (воркера)."""
    import knowledge_mcp.main as main_mod
    try:
        # Проверяем, есть ли ссылка на сохраненный Event Loop главного потока
        if hasattr(main_mod, "event_loop") and main_mod.event_loop:
            asyncio.run_coroutine_threadsafe(manager.broadcast(message), main_mod.event_loop)
        else:
            # Fallback для тестов
            try:
                asyncio.get_running_loop()
                asyncio.create_task(manager.broadcast(message))
            except RuntimeError:
                pass
    except Exception as e:
        logger.error(f"Failed to broadcast message: {e}")

