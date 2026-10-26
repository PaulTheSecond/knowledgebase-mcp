import argparse
import os
import sys
import logging
from contextlib import asynccontextmanager
import uvicorn
import asyncio
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла в корне проекта
load_dotenv()
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, WebSocket, WebSocketDisconnect, Header
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional

from knowledge_mcp.db.models import KnowledgeDB
from knowledge_mcp.daemon.auth import verify_basic_auth, generate_temp_token
from knowledge_mcp.core.i18n import t, get_locale_from_header

# НЕ импортируем Indexer на верхнем уровне!
# Indexer -> embeddings.py -> sentence_transformers -> torch (~30 сек импорт).
# Для MCP-команды Indexer не нужен вообще, поэтому импортируем лениво.

# ВАЖНО: Весь логгер должен идти в sys.stderr, иначе MCP (общающийся по stdout) сломается!
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

event_loop = None
start_daemon = False
worker_instance = None
watcher_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global event_loop, worker_instance
    event_loop = asyncio.get_running_loop()
    
    scheduler = None
    if db:
        stale_count = db.reset_stale_jobs()
        if stale_count > 0:
            logger.info(f"Reset {stale_count} stale processing job(s) on server startup.")
        # Предварительный фоновый прогрев кэша статистики
        asyncio.create_task(asyncio.to_thread(db.get_all_repo_stats, True))
        asyncio.create_task(asyncio.to_thread(db.get_cas_stats, None, True))

    if db and start_daemon:
        import threading
        from knowledge_mcp.daemon.worker import DaemonWorker
        from knowledge_mcp.daemon.scheduler import CronScheduler
        
        worker_instance = DaemonWorker(db.db_path)
        worker_thread = threading.Thread(target=worker_instance.start, daemon=True, name="DaemonWorkerThread")
        worker_thread.start()
        logger.info("DaemonWorker background thread started.")
        
        scheduler = CronScheduler(db)
        scheduler.start()
        logger.info("CronScheduler background timers started.")

    try:
        yield
    finally:
        if scheduler:
            scheduler.stop()
            logger.info("CronScheduler stopped.")
        if worker_instance:
            worker_instance.stop()
            logger.info("DaemonWorker stopped.")
            worker_instance = None
        event_loop = None

app = FastAPI(title="Knowledge Base Sync Trigger", lifespan=lifespan)

from knowledge_mcp.daemon.routes.chat import router as chat_router
from knowledge_mcp.daemon.routes.settings import router as settings_router

app.include_router(chat_router)
app.include_router(settings_router)
from knowledge_mcp.daemon.routes.telemetry import router as telemetry_router
app.include_router(telemetry_router)

@app.websocket("/ws/status")
async def websocket_status_endpoint(websocket: WebSocket):
    """WebSocket канал для обновления статуса синхронизации в реальном времени."""
    token = websocket.query_params.get("token")
    from knowledge_mcp.daemon.auth import verify_temp_token
    if not token or not verify_temp_token(token):
        await websocket.close(code=4001)
        return
        
    from knowledge_mcp.daemon.status_ws import manager
    await manager.connect(websocket)
    try:
        while True:
            # Ожидаем данных от клиента, чтобы держать соединение открытым
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

db = None
indexer = None

class SyncRequest(BaseModel):
    repo_id: str
    repo_path: str
    force: bool = False

def init_components(db_path: str):
    global db, indexer
    from knowledge_mcp.core.indexer import Indexer  # Ленивый импорт — только для sync/serve
    from knowledge_mcp.core.config import load_config
    cfg = load_config()
    use_emb = cfg.get("embeddings", {}).get("enabled", False)
    db = KnowledgeDB(db_path)
    indexer = Indexer(db, use_embeddings=use_emb)
    
    # Синхронизируем глобальные переменные для обоих вариантов загрузки (__main__ и knowledge_mcp.main)
    import sys
    for mod_name in ("knowledge_mcp.main", "__main__"):
        mod = sys.modules.get(mod_name)
        if mod:
            mod.db = db
            mod.indexer = indexer

    # Инициализируем и подключаем роутер вебхуков (Story 1.5)
    from knowledge_mcp.daemon.routes.webhooks import init_webhook_router
    app.include_router(init_webhook_router(db))


@app.post("/sync")
async def sync_endpoint(req: SyncRequest, background_tasks: BackgroundTasks):
    """
    HTTP/Webhook Триггер. 
    Выполняет дельта-скан в фоне, не блокируя ответ планировщику (cron/github hooks).
    """
    if not db or not indexer:
        raise HTTPException(status_code=500, detail="Components not initialized")
    
    background_tasks.add_task(indexer.sync_repo, req.repo_id, req.repo_path, force=req.force)
    return {"status": "Sync queued", "repo_id": req.repo_id, "background": True}

@app.post("/api/auth/token")
async def get_token(username: str = Depends(verify_basic_auth)):
    """Выпускает временный короткоживущий токен для WebSockets/SSE."""
    token = generate_temp_token(username)
    return {"token": token}

def _compute_status_data(db_instance: KnowledgeDB) -> dict:
    row = db_instance.get_system_status()
    all_stats = db_instance.get_all_repo_stats()
    
    from knowledge_mcp.core.config import load_config
    config = load_config()
    repos_data = []
    for repo in config.get("repositories", []):
        repo_id = repo.get("id")
        stats = all_stats.get((repo_id, None)) or all_stats.get((repo_id, "default")) or {"files_count": 0, "chunks_count": 0, "vectors_count": 0}
        repos_data.append({
            "id": repo_id,
            "url": repo.get("url") or repo.get("local_path", ""),
            "branch": repo.get("branch", "master"),
            "files_count": stats["files_count"],
            "chunks_count": stats["chunks_count"],
            "vectors_count": stats["vectors_count"]
        })

    # Проектные воркспейсы (Multi-repo Worktrees / v1.8)
    for project in config.get("projects", []):
        proj_id = project.get("id")
        ws_root = project.get("workspace_root", "")
        base_path = project.get("base_path", "worktrees/master")
        for p_repo in project.get("repositories", []):
            r_id = p_repo.get("id")
            stats = all_stats.get((r_id, proj_id)) or {"files_count": 0, "chunks_count": 0, "vectors_count": 0}
            branch = "master" if r_id != "browser-extension" else "main"
            repos_data.append({
                "id": f"{proj_id}/{r_id}",
                "url": f"{ws_root}/{base_path}/{p_repo.get('path', r_id)}",
                "branch": branch,
                "files_count": stats["files_count"],
                "chunks_count": stats["chunks_count"],
                "vectors_count": stats["vectors_count"]
            })
        doc_stats = all_stats.get(("docs", proj_id)) or {"files_count": 0, "chunks_count": 0, "vectors_count": 0}
        if doc_stats["files_count"] > 0:
            repos_data.append({
                "id": f"{proj_id}/docs",
                "url": f"{ws_root} (Docs & .kb)",
                "branch": "master",
                "files_count": doc_stats["files_count"],
                "chunks_count": doc_stats["chunks_count"],
                "vectors_count": doc_stats["vectors_count"]
            })

    sys_status = "idle"
    last_error = None
    last_indexed_at = None
    if row:
        sys_status = row["status"]
        last_error = row["last_error"]
        last_indexed_at = row["last_indexed_at"]

    chat_enabled = config.get("server", {}).get("dashboard", {}).get("chat", {}).get("enabled", False)

    return {
        "status": sys_status,
        "last_error": last_error,
        "last_indexed_at": last_indexed_at,
        "repositories": repos_data,
        "chat_enabled": chat_enabled
    }

@app.get("/api/status")
async def get_status(username: str = Depends(verify_basic_auth)):
    """Возвращает статус системы и расширенную телеметрию репозиториев (non-blocking)."""
    if not db:
        raise HTTPException(status_code=500, detail="Components not initialized")
    return await asyncio.to_thread(_compute_status_data, db)

@app.get("/api/queue")
async def api_get_queue(username: str = Depends(verify_basic_auth)):
    """Возвращает последние 50 задач из очереди синхронизации (non-blocking)."""
    if not db:
        raise HTTPException(status_code=500, detail="Components not initialized")
    jobs = await asyncio.to_thread(db.get_queue_jobs, 50)
    return {"jobs": jobs}

def _get_request_locale(accept_language: Optional[str] = None) -> str:
    default_loc = db.get_setting("default_locale", "ru") if db else "ru"
    return get_locale_from_header(accept_language, default_loc)

@app.post("/api/queue/retry-failed")
async def api_retry_failed_jobs(
    username: str = Depends(verify_basic_auth),
    accept_language: Optional[str] = Header(None, alias="Accept-Language")
):
    """Сбрасывает все упавшие задачи очереди обратно в pending для повторной попытки."""
    locale = _get_request_locale(accept_language)
    if not db:
        raise HTTPException(status_code=500, detail=t("system.not_initialized", locale=locale))
    count = db.retry_failed_jobs()
    return {
        "status": "retried",
        "count": count,
        "message": t("queue.retried", locale=locale, count=count)
    }

@app.post("/api/queue/{job_id}/stop")
async def api_stop_job(
    job_id: int,
    username: str = Depends(verify_basic_auth),
    accept_language: Optional[str] = Header(None, alias="Accept-Language")
):
    """Останавливает задачу в очереди."""
    locale = _get_request_locale(accept_language)
    if not db:
        raise HTTPException(status_code=500, detail=t("system.not_initialized", locale=locale))

    if worker_instance:
        worker_instance.cancel_active_job(job_id)

    success = db.stop_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail=t("queue.job_cannot_stop", locale=locale, job_id=job_id))
    return {
        "status": "stopped",
        "job_id": job_id,
        "message": t("queue.job_stopped", locale=locale, job_id=job_id)
    }

@app.post("/api/queue/{job_id}/restart")
async def api_restart_job(
    job_id: int,
    username: str = Depends(verify_basic_auth),
    accept_language: Optional[str] = Header(None, alias="Accept-Language")
):
    """Перезапускает задачу в очереди (сбрасывает в pending)."""
    locale = _get_request_locale(accept_language)
    if not db:
        raise HTTPException(status_code=500, detail=t("system.not_initialized", locale=locale))

    success = db.restart_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=t("queue.job_not_found", locale=locale, job_id=job_id))
    return {
        "status": "restarted",
        "job_id": job_id,
        "message": t("queue.job_restarted", locale=locale, job_id=job_id)
    }

@app.delete("/api/queue/{job_id}")
async def api_delete_job(
    job_id: int,
    username: str = Depends(verify_basic_auth),
    accept_language: Optional[str] = Header(None, alias="Accept-Language")
):
    """Удаляет задачу из очереди."""
    locale = _get_request_locale(accept_language)
    if not db:
        raise HTTPException(status_code=500, detail=t("system.not_initialized", locale=locale))

    if worker_instance:
        worker_instance.cancel_active_job(job_id)

    success = db.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=t("queue.job_not_found", locale=locale, job_id=job_id))
    return {
        "status": "deleted",
        "job_id": job_id,
        "message": t("queue.job_deleted", locale=locale, job_id=job_id)
    }

@app.post("/api/sync/{repo_id}")
async def api_trigger_sync(
    repo_id: str,
    username: str = Depends(verify_basic_auth),
    accept_language: Optional[str] = Header(None, alias="Accept-Language")
):
    """Ставит задачу синхронизации в очередь."""
    locale = _get_request_locale(accept_language)
    if not db:
        raise HTTPException(status_code=500, detail=t("system.not_initialized", locale=locale))
    job_id = db.add_to_queue(repo_id)
    return {
        "status": "queued",
        "repo_id": repo_id,
        "job_id": job_id,
        "message": t("queue.sync_queued", locale=locale, repo_id=repo_id, job_id=job_id)
    }

@app.delete("/api/repo/{repo_id}")
async def api_delete_repo(
    repo_id: str,
    username: str = Depends(verify_basic_auth),
    accept_language: Optional[str] = Header(None, alias="Accept-Language")
):
    """Удаляет все данные репозитория из индекса."""
    locale = _get_request_locale(accept_language)
    if not db:
        raise HTTPException(status_code=500, detail=t("system.not_initialized", locale=locale))
    deleted_files = db.delete_repo(repo_id)
    return {
        "status": "deleted",
        "repo_id": repo_id,
        "files_deleted": deleted_files,
        "message": t("repo.deleted", locale=locale, repo_id=repo_id)
    }

# Раздача Frontend Dashboard (Angular SPA или Fallback Static)
angular_dist_dir = Path(__file__).resolve().parent.parent / "frontend" / "dist" / "dashboard" / "browser"

@app.get("/")
@app.get("/{full_path:path}")
async def serve_spa_or_static(full_path: str = ""):
    """
    Раздает статические файлы и SPA index.html для роутинга Angular.
    """
    # Не перехватываем обращения к API или WebSocket
    if full_path.startswith("api/") or full_path.startswith("ws/") or full_path == "sync":
        raise HTTPException(status_code=404, detail="Not Found")

    if angular_dist_dir.exists():
        # Если запрошен конкретный файл бандла (например, main-xxx.js, styles-xxx.css, favicon.ico)
        if full_path:
            file_path = (angular_dist_dir / full_path).resolve()
            if file_path.is_file() and str(file_path).startswith(str(angular_dist_dir.resolve())):
                return FileResponse(file_path)

        # Для маршрутов SPA (/, /repositories, /queue, /chat) отдаем index.html
        index_file = angular_dist_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)

    raise HTTPException(status_code=404, detail="Dashboard index.html not found. Please build frontend first.")

def main():
    if sys.platform == "win32":
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    def get_effective_db_path() -> str:
        db_env = os.environ.get("DB_PATH", "data/knowledge.db")
        if sys.platform == "win32" and db_env.startswith("/data/") and not Path(db_env).exists():
            candidate = Path(db_env.lstrip("/"))
            if candidate.exists() or Path("data").exists():
                return str(candidate)
        return db_env

    parser = argparse.ArgumentParser(description="Knowledge MCP - Indexer & Trigger Server")
    # Дефолт берётся из переменной окружения DB_PATH (Docker задаёт её автоматически),
    # а без неё — data/knowledge.db для локального запуска
    parser.add_argument("--db-path", type=str,
                        default=get_effective_db_path(),
                        help="Path to SQLite DB (default: $DB_PATH or data/knowledge.db)")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Режим ручного запуска (CLI-команда)
    sync_parser = subparsers.add_parser("sync", help="Run strict delta-sync for repository (CLI trigger)")
    sync_parser.add_argument("--repo-id", required=True, type=str, help="Repository ID")
    sync_parser.add_argument("--repo-path", required=True, type=str, help="Path to repository root")
    sync_parser.add_argument("--all", action="store_true", help="Index all directories without interactive prompt")
    sync_parser.add_argument("--force", action="store_true", help="Force re-indexing of all files ignoring mtime/hash cache")
    
    # Режим сервера (HTTP endpoint)
    serve_parser = subparsers.add_parser("serve", help="Start HTTP server to listen for /sync webhook triggers")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    
    # Режим MCP-сервера (Codex, Claude, Gemini)
    # По умолчанию эмбеддинги ОТКЛЮЧЕНЫ для мгновенного старта.
    # Новая модель (e5-small) в 2-3 раза быстрее и компактнее старой.
    mcp_parser = subparsers.add_parser("mcp", help="Start the MCP stdio server")
    mcp_parser.add_argument("--with-embeddings", action="store_true",
                            help="Enable vector search (loads ~150MB model, high quality RAG)")

    # Режим генерации интеграционных пресетов для AI-агентов (Claude, Antigravity, Cursor, Codex)
    preset_parser = subparsers.add_parser("preset", help="Generate or display integration presets for AI agents")
    preset_parser.add_argument("agent", nargs="?", default="all",
                               choices=["claude", "antigravity", "cursor", "codex", "all"],
                               help="Target AI agent (default: all)")
    preset_parser.add_argument("--output-dir", type=str, default=None,
                               help="Directory to save preset files (if omitted, snippets are printed to stdout)")

    # Режим быстрой синхронизации одного файла (хуки PostToolUse)
    sync_file_parser = subparsers.add_parser("sync-file", help="JIT index or check single file for agent hooks")
    sync_file_parser.add_argument("--file", type=str, default=None, help="Path to modified or created file")
    sync_file_parser.add_argument("--project-id", type=str, default=None, help="Optional project ID override")
    sync_file_parser.add_argument("--repo-id", type=str, default=None, help="Optional repo ID override")
    sync_file_parser.add_argument("--branch", type=str, default=None, help="Optional branch override")
    sync_file_parser.add_argument("--check-status", action="store_true", help="Quickly verify DB status and active workspaces")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "preset":
        from knowledge_mcp.core.agent_presets import AgentPresetManager
        if args.output_dir:
            created = AgentPresetManager.export_presets(args.agent, args.output_dir)
            print(f"[KnowledgeBase MCP] Успешно создано {len(created)} файл(ов) пресета в '{args.output_dir}':")
            for f in created:
                print(f"  - {f}")
        else:
            AgentPresetManager.print_presets(args.agent)
        return
    elif args.command == "sync-file":
        db_path = getattr(args, "db_path", os.environ.get("DB_PATH", "data/knowledge.db"))
        db = KnowledgeDB(db_path)

        if getattr(args, "check_status", False):
            from knowledge_mcp.core.config import load_config, get_projects
            cfg = load_config()
            repos_count = len(cfg.get("repositories", []))
            projects_count = len(get_projects())

            cursor = db.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM files")
            files_count = cursor.fetchone()[0]

            print("[KnowledgeBase MCP] Status: OK")
            print(f"Database: {db_path} (accessible)")
            print(f"Workspaces: {projects_count} project(s), {repos_count} repository(ies)")
            print(f"Indexed files in CAS: {files_count}")
            db.close()
            return

        if not getattr(args, "file", None):
            logger.error("Требуется указать --file <путь> или --check-status")
            sys.exit(1)

        target_file = Path(args.file).resolve()
        from knowledge_mcp.core.workspace_scanner import WorkspaceScanner
        from knowledge_mcp.core.jit_sync import JITSyncEngine
        from knowledge_mcp.core.indexer import Indexer

        project_id = getattr(args, "project_id", None)
        repo_id = getattr(args, "repo_id", None)
        branch = getattr(args, "branch", None)
        repo_root = None

        if not (project_id and repo_id and branch):
            discovered = WorkspaceScanner.resolve_active_target(target_file)
            if discovered:
                project_id = project_id or discovered.project_id
                repo_id = repo_id or discovered.repo_id
                branch = branch or discovered.branch
                repo_root = discovered.absolute_path

        if not repo_root:
            curr = target_file.parent if target_file.is_file() else target_file
            while curr != curr.parent:
                if (curr / ".git").exists():
                    repo_root = curr
                    break
                curr = curr.parent

            if repo_root:
                repo_id = repo_id or repo_root.name
                branch = branch or WorkspaceScanner.get_git_branch(repo_root)
                project_id = project_id or "default"
            else:
                repo_root = target_file.parent
                repo_id = repo_id or "default"
                branch = branch or "master"
                project_id = project_id or "default"

        fast_indexer = Indexer(db, use_embeddings=False)
        jit = JITSyncEngine(fast_indexer)
        ok = jit.index_single_file(
            target_file,
            project_id=project_id,
            repo_id=repo_id,
            branch=branch,
            db=db,
            repo_root=repo_root
        )
        if ok:
            action = "Indexed" if target_file.exists() else "Deleted"
            print(f"[JIT] {action} {target_file.name} in repo '{repo_id}' (branch: '{branch}', project: '{project_id}')")
        else:
            logger.warning(f"[JIT] Could not index single file: {target_file}")
            db.close()
            sys.exit(1)
        db.close()
        return
    elif args.command == "mcp":
        # MCP-сервер имеет собственную лёгкую инициализацию (без Indexer/LocalEmbedder),
        # чтобы handshake проходил мгновенно и не блокировался загрузкой ML-модели.
        import asyncio
        from knowledge_mcp.mcp.server import start_mcp_server
        
        # Проверка: если БД не существует, MCP сервер может запуститься вхолостую
        if not os.path.exists(args.db_path):
            logger.error(f"Database file not found at {args.db_path}. Please check your volume mounts.")
            # Мы не выходим с ошибкой, так как SQLite создаст пустую БД, 
            # но мы хотя бы предупреждаем в stderr.
            
        use_emb = getattr(args, 'with_embeddings', False)
        asyncio.run(start_mcp_server(args.db_path, enable_embeddings=use_emb))
    else:
        # Для sync и serve нужен полный стек (включая Indexer с эмбеддингами)
        init_components(args.db_path)

        if args.command == "sync":
            allowed_top_level = None
            is_all = getattr(args, "all", False)
            force = getattr(args, "force", False)

            if not is_all:
                if not sys.stdin.isatty():
                    logger.warning("Неинтерактивный терминал (отсутствует TTY). Автоматически включен пакетный режим (--all).")
                    is_all = True
                else:
                    import questionary
                    repo_path_obj = Path(args.repo_path).resolve()
                    if not repo_path_obj.exists():
                        logger.error(f"Path does not exist: {repo_path_obj}")
                        sys.exit(1)
                    
                    spec = indexer._get_ignore_spec(repo_path_obj)
                    
                    # Собираем элементы первого уровня
                    choices = []
                    for entry in os.scandir(repo_path_obj):
                        rel_p = entry.name + ('/' if entry.is_dir() else '')
                        if not spec.match_file(rel_p):
                            choices.append(entry.name)
                    
                    if not choices:
                        logger.warning("No unignored files/directories found at top level.")
                    else:
                        selected = questionary.checkbox(
                            "Выберите папки/файлы для индексации (Enter - выбрать все отмеченные):",
                            choices=[questionary.Choice(c, checked=True) for c in sorted(choices)]
                        ).ask()
                        
                        if selected is None:
                            logger.info("Sync cancelled by user.")
                            sys.exit(0)
                            
                        allowed_top_level = selected

            logger.info(f"Manual CLI trigger logic for {args.repo_id}")
            indexer.sync_repo(args.repo_id, args.repo_path, allowed_top_level=allowed_top_level, force=force)
        elif args.command == "serve":
            global start_daemon
            start_daemon = True
            logger.info(f"Starting trigger HTTP endpoint on {args.host}:{args.port}")
            uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
