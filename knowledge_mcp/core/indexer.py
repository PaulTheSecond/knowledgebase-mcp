import os
import gc
import hashlib
import json
import logging
import subprocess
import uuid
from enum import Enum
from pathlib import Path
from typing import Generator, Optional, List, Dict, Any, Tuple
import concurrent.futures

import pathspec

from knowledge_mcp.db.models import KnowledgeDB
from .embeddings import LocalEmbedder
from .code_parser import CodeParser, LANGUAGE_MAP
from .markdown_parser import MarkdownParser

logger = logging.getLogger(__name__)


def hash_file(filepath: Path, chunk_size: int = 8192) -> str:
    """Вычисляет SHA-256 хэш содержимого файла."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


# Размер суб-батча для обработки эмбеддингов порциями (оптимизировано для баланса RAM/скорость)
EMBED_SUB_BATCH_SIZE = 128
# Максимальное количество потоков для параллельного хэширования
MAX_HASH_WORKERS = 8
# Количество файлов в одном батче транзакции SQLite для предотвращения переполнения буферов и disk I/O error
BATCH_COMMIT_SIZE = 50


class ReindexStatus(Enum):
    """Результат попытки реиндексации одного файла."""
    QUEUED_FOR_EMBEDDING = "queued"       # Чанки добавлены в очередь на векторизацию
    PROCESSED_NO_EMBEDDING = "processed"  # Файл обработан успешно, эмбеддинги не нужны
    PARSE_ERROR = "error"                 # Файл пропущен из-за ошибки (бинарник, кодировка)


class Indexer:
    def __init__(self, db: KnowledgeDB, use_embeddings: bool = True):
        self.db = db
        self.use_embeddings = use_embeddings
        self.embedder = LocalEmbedder() if use_embeddings else None
        # Кэш результатов Roslyn: resolved Path -> {'symbols': [...], 'edges': [...]}
        self.csharp_cache: dict = {}
        # Переиспользуемые инстансы парсеров (Tree-sitter инициализируется один раз)
        self.code_parser = CodeParser()
        self.md_parser = MarkdownParser()

    # ──────────────────────────────────────────────────────────────────────
    # Вспомогательные методы: обход файлов
    # ──────────────────────────────────────────────────────────────────────

    def _get_ignore_spec(self, root_path: Path) -> pathspec.PathSpec:
        """Считывает .gitignore и добавляет системные исключения."""
        patterns = [
            # Системные
            '.git/', 'node_modules/', '.idea/', '.vs/', 'venv/', '__pycache__/', '*.log', '*.db',
            # AI-агенты и их конфиги
            '_bmad/', '.agents/', '.ai/', '.bmad-core/', '.claude/', '.gemini/',
            '.opencode/', '.roo/', '.github/', '.nuget/',
            'mcp.json', 'opencode.json', '.task-cache.json',
            'gitlab-task-collector.config.json',
            # Конфиги сборки и IDE
            '.editorconfig', '.arscontexta', '.mailmap', '.roomodes',
            'Directory.Build.props', 'Directory.Packages.props',
            'nuget.config', 'packages.txt',
        ]

        gitignore_path = root_path / '.gitignore'
        if gitignore_path.exists():
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                patterns.extend(f.readlines())

        return pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, patterns)

    def _walk_files(self, root_path: Path, spec: pathspec.PathSpec, allowed_top_level: list[str] = None) -> Generator[Path, None, None]:
        """Генератор, рекурсивно обходящий файлы с исключением путей по gitignore."""
        for dirpath, dirnames, filenames in os.walk(root_path):
            dir_rel_path = os.path.relpath(dirpath, root_path)
            if dir_rel_path == '.':
                dir_rel_path = ''

            # Если мы в корне и задан список top-level, оставляем только их
            if dir_rel_path == '' and allowed_top_level is not None:
                dirnames[:] = [d for d in dirnames if d in allowed_top_level]
                filenames = [f for f in filenames if f in allowed_top_level]

            # Исключаем директории, соответствующие паттернам, чтобы os.walk внутрь не заходил
            dirnames[:] = [d for d in dirnames if not spec.match_file((os.path.join(dir_rel_path, d) + '/').replace('\\', '/'))]

            for f in filenames:
                file_rel_path = os.path.join(dir_rel_path, f).replace('\\', '/')
                if not spec.match_file(file_rel_path):
                    yield Path(dirpath) / f

    # ──────────────────────────────────────────────────────────────────────
    # Roslyn: семантический анализ .NET проектов
    # ──────────────────────────────────────────────────────────────────────

    def _run_roslyn_analysis(self, repo_path: Path, spec: pathspec.PathSpec):
        """
        Запускает RoslynParser для .NET проектов и заполняет self.csharp_cache.
        Кэш содержит символы и рёбра для каждого .cs файла.
        """
        self.csharp_cache = {}

        # Оптимизированный поиск .sln файлов с уважением к .gitignore
        csharp_projects = [f for f in self._walk_files(repo_path, spec) if f.suffix == '.sln']
        if not csharp_projects:
            csharp_projects = [f for f in self._walk_files(repo_path, spec) if f.suffix == '.csproj']

        if not csharp_projects:
            return

        # Кэш path.resolve() для избежания миллионов обращений к диску по Docker Volume.
        # ВАЖНО: объявляем ВНЕ цикла по проектам, чтобы кэш не сбрасывался на каждой итерации.
        path_resolved_cache: dict[str, Path] = {}

        def get_resolved(path_str: str) -> Path:
            if path_str not in path_resolved_cache:
                path_resolved_cache[path_str] = Path(path_str).resolve()
            return path_resolved_cache[path_str]

        app_root = Path(__file__).resolve().parents[2]
        parser_dll = app_root / "RoslynParser" / "bin" / "Release" / "net8.0" / "RoslynParser.dll"
        parser_csproj = app_root / "RoslynParser" / "RoslynParser.csproj"

        for proj in csharp_projects:
            if "RoslynParser" in str(proj):
                continue

            logger.info(f"Running Roslyn semantic analysis on {proj}...")

            if parser_dll.exists():
                cmd = ["dotnet", str(parser_dll), str(proj)]
            else:
                cmd = [
                    "dotnet", "run", "-c", "Release",
                    "--project", str(parser_csproj),
                    "--", str(proj)
                ]

            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                logger.error(f"Roslyn analysis failed for {proj}: {res.stderr}")
                continue

            try:
                data = json.loads(res.stdout)

                for s in data.get('symbols', []):
                    fp = get_resolved(s['file_path'])
                    if fp not in self.csharp_cache:
                        self.csharp_cache[fp] = {'symbols': [], 'edges': []}
                    self.csharp_cache[fp]['symbols'].append(s)

                id_to_file = {
                    s['ast_node_id']: get_resolved(s['file_path'])
                    for s in data.get('symbols', [])
                }

                for e in data.get('edges', []):
                    if e['source_ast_id'] in id_to_file:
                        fp = id_to_file[e['source_ast_id']]
                        self.csharp_cache[fp]['edges'].append(e)

            except Exception as ex:
                logger.error(f"Failed to parse Roslyn output for {proj}: {ex}")

    # ──────────────────────────────────────────────────────────────────────
    # Общий helper: создание чанка + постановка в очередь эмбеддингов
    # ──────────────────────────────────────────────────────────────────────

    def _add_chunk_with_embedding(
        self,
        file_id: int,
        rel_path: str,
        content: str,
        source_kind: str,
        trust: str,
        line_start: int,
        line_end: int,
        chunks_to_embed: list,
    ) -> tuple[str, int]:
        """
        Создаёт чанк в БД и при необходимости добавляет его в очередь на векторизацию.
        Возвращает (chunk_id, chunk_rowid).
        Централизует дублирующийся паттерн uuid+add_chunk+append из 4 ветвей _reindex_file.
        """
        chunk_id = str(uuid.uuid4())
        chunk_rowid = self.db.add_chunk(
            chunk_id=chunk_id, file_id=file_id, content=content,
            source_kind=source_kind, trust=trust,
            line_start=line_start, line_end=line_end,
        )
        if self.use_embeddings and self.embedder:
            chunks_to_embed.append((content, chunk_rowid, rel_path))
        return chunk_id, chunk_rowid

    # ──────────────────────────────────────────────────────────────────────
    # Слой 1: Чистые функции парсинга (без доступа к БД)
    # Безопасны для запуска в ThreadPoolExecutor.
    # Возвращают tuple: (chunks_raw, symbols_raw, edges_raw)
    #   chunks_raw:  [(body, source_kind, trust, line_start, line_end), ...]
    #   symbols_raw: [(name, qname, kind, lang, ls, le, sig, body_idx), ...]
    #                body_idx — индекс в chunks_raw, которому принадлежит символ
    #   edges_raw:   [(source_qname_or_ast_id, target_name, kind, is_ast_id), ...]
    # ──────────────────────────────────────────────────────────────────────

    def _parse_csharp(self, filepath: Path) -> tuple[list, list, list]:
        """Парсинг .cs файла из кэша Roslyn (чистая функция, нет обращений к БД)."""
        cache_hit = self.csharp_cache.get(filepath.resolve())
        if not cache_hit:
            return [], [], []

        chunks_raw = []
        symbols_raw = []
        edges_raw = []

        ast_id_to_chunk_idx: dict = {}
        for s in cache_hit['symbols']:
            idx = len(chunks_raw)
            chunks_raw.append((s['body'], 'code', 'verified', s['line_start'], s['line_end']))
            symbols_raw.append((
                s['name'], s['qualified_name'], s['kind'], s['language'],
                s['line_start'], s['line_end'], s['signature'], idx
            ))
            ast_id_to_chunk_idx[s['ast_node_id']] = idx

        for e in cache_hit['edges']:
            if e['source_ast_id'] in ast_id_to_chunk_idx:
                edges_raw.append((e['source_ast_id'], e['target_qualified_name'], e['kind'], True))

        return chunks_raw, symbols_raw, edges_raw

    def _parse_code(self, filepath: Path) -> tuple[list, list, list]:
        """Парсинг кодового файла через Tree-sitter (чистая функция, нет обращений к БД)."""
        lang = LANGUAGE_MAP[filepath.suffix.lower()]
        symbols = self.code_parser.parse_file(filepath, lang)
        edges_ts = self.code_parser.extract_edges(filepath, lang)

        chunks_raw = []
        symbols_raw = []
        edges_raw = []
        qname_to_idx: dict = {}

        for symbol in symbols:
            if not symbol.body.strip():
                continue
            idx = len(chunks_raw)
            chunks_raw.append((symbol.body, 'code', 'verified', symbol.line_start, symbol.line_end))
            symbols_raw.append((
                symbol.name, symbol.qualified_name, symbol.kind, symbol.language,
                symbol.line_start, symbol.line_end, symbol.signature, idx
            ))
            qname_to_idx[symbol.qualified_name] = idx
            if 'file_level' not in qname_to_idx:
                qname_to_idx['file_level'] = idx

        for edge in edges_ts:
            edges_raw.append((edge.source_name, edge.target_name, edge.kind, False))

        return chunks_raw, symbols_raw, edges_raw

    @staticmethod
    def _detect_language(filepath: Path) -> str:
        """Определяет язык файла по расширению."""
        ext = filepath.suffix.lower()
        mapping = {
            '.cs': 'c_sharp',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.py': 'python',
            '.sql': 'sql',
            '.md': 'markdown',
            '.mdx': 'markdown',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
        }
        return mapping.get(ext, 'text')

    def _parse_markdown(self, filepath: Path, rel_path: str = "") -> tuple[list, list, list]:
        """Парсинг Markdown (.kb/, CLAUDE.md, docs) с разметкой source_kind='rules' или 'documentation'."""
        sections = self.md_parser.parse_file(filepath)
        fname = filepath.name.lower()
        norm_rel = rel_path.replace('\\', '/').lower()
        if fname in ('claude.md', '.cursorrules') or 'rules' in fname or norm_rel.startswith('.kb/') or '/.kb/' in norm_rel:
            source_kind = 'rules'
            trust = 'verified'
        else:
            source_kind = 'documentation'
            trust = 'hint'

        chunks_raw = [
            (sec.content, source_kind, trust, sec.line_start, sec.line_end)
            for sec in sections if sec.content.strip()
        ]
        if not chunks_raw:
            try:
                content = filepath.read_text(encoding='utf-8', errors='replace')
                if content.strip():
                    lines_count = len(content.splitlines())
                    chunks_raw = [(content, source_kind, trust, 1, lines_count)]
            except Exception:
                pass
        return chunks_raw, [], []

    def _parse_fallback(self, filepath: Path) -> tuple[list, list, list]:
        """Fallback: читает файл целиком (чистая функция, нет обращений к БД)."""
        content = filepath.read_text(encoding='utf-8', errors='replace')
        if not content.strip():
            return [], [], []
        lines_count = len(content.splitlines())
        return [(content, 'code', 'verified', 1, lines_count)], [], []

    def _parse_file_pure(self, filepath: Path, rel_path: str) -> tuple[list, list, list] | None:
        """
        Диспетчер чистого парсинга файла.
        Возвращает (chunks_raw, symbols_raw, edges_raw) или None при ошибке.
        Безопасен для вызова из ThreadPoolExecutor.
        """
        ext = filepath.suffix.lower()
        fname = filepath.name.lower()
        norm_rel = rel_path.replace('\\', '/').lower()
        try:
            if ext in ('.md', '.mdx') or norm_rel.startswith(('docs/', 'knowledge/', '.kb/')) or '/.kb/' in norm_rel or fname in ('claude.md', '.cursorrules'):
                return self._parse_markdown(filepath, rel_path)
            elif ext == '.cs':
                res = self._parse_csharp(filepath)
                if res and (res[0] or res[1]):
                    return res
                # Fallback на Tree-sitter парсер, если Roslyn кэш пуст
                return self._parse_code(filepath)
            elif ext in LANGUAGE_MAP:
                return self._parse_code(filepath)
            else:
                return self._parse_fallback(filepath)
        except Exception as e:
            logger.warning(f"Failed parsing file {rel_path}: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────
    # Слой 2: Батчевая запись результатов парсинга в БД
    # ──────────────────────────────────────────────────────────────────────

    def _flush_parsed_file(
        self,
        file_id: int,
        rel_path: str,
        parsed: tuple[list, list, list],
        chunks_to_embed: list,
        content_hash: Optional[str] = None
    ) -> ReindexStatus:
        """
        Записывает результаты парсинга в БД батчевыми операциями с поддержкой content_hash и TESTED_BY.
        Возвращает ReindexStatus.
        """
        chunks_raw, symbols_raw, edges_raw = parsed

        if not chunks_raw and not symbols_raw:
            return ReindexStatus.PROCESSED_NO_EMBEDDING

        if content_hash is None and file_id:
            cursor = self.db.conn.cursor()
            cursor.execute("SELECT content_hash FROM files WHERE id = ?", (file_id,))
            row = cursor.fetchone()
            if row and row['content_hash']:
                content_hash = row['content_hash']

        # --- Батчевая вставка чанков с content_hash ---
        chunk_records = [
            (str(uuid.uuid4()), file_id, body, ls, le, sk, tr, None, content_hash)
            for body, sk, tr, ls, le in chunks_raw
        ]
        rowids = self.db.add_chunks_batch(chunk_records)
        chunk_ids = [r[0] for r in chunk_records]

        # Добавляем в очередь эмбеддингов
        embed_added = False
        if self.use_embeddings and self.embedder:
            for i, (body, sk, tr, ls, le) in enumerate(chunks_raw):
                chunks_to_embed.append((body, rowids[i], rel_path))
                embed_added = True

        # --- Батчевая вставка символов с content_hash ---
        if symbols_raw:
            sym_records = [
                (file_id, name, qname, kind, lang, ls, le, sig, chunk_ids[body_idx], content_hash)
                for name, qname, kind, lang, ls, le, sig, body_idx in symbols_raw
            ]
            sym_rowids = self.db.add_symbols_batch(sym_records)
            qname_to_sym_id: dict = {}
            ast_to_sym_id: dict = {}
            for i, (name, qname, kind, lang, ls, le, sig, body_idx) in enumerate(symbols_raw):
                qname_to_sym_id[qname] = sym_rowids[i]
                if name not in qname_to_sym_id:
                    qname_to_sym_id[name] = sym_rowids[i]
                if not qname_to_sym_id.get('file_level'):
                    qname_to_sym_id['file_level'] = sym_rowids[i]

            if edges_raw and edges_raw[0][3]:  # is_ast_id
                resolved_key = None
                for rk, cv in self.csharp_cache.items():
                    if str(rk).replace('\\', '/').endswith(rel_path):
                        resolved_key = rk
                        break
                if resolved_key:
                    cache_syms = self.csharp_cache[resolved_key]['symbols']
                    for i, s in enumerate(cache_syms):
                        if i < len(sym_rowids):
                            ast_to_sym_id[s['ast_node_id']] = sym_rowids[i]

            # --- Батчевая вставка рёбер ---
            edge_records = []
            unresolved_records = []
            for src_key, target_name, kind, is_ast_id in edges_raw:
                if kind == 'TESTED_BY':
                    # Направление TESTED_BY: source = протестированный символ, target = тест
                    test_id = qname_to_sym_id.get(target_name) or (ast_to_sym_id.get(target_name) if is_ast_id else None)
                    if not test_id:
                        test_id = qname_to_sym_id.get(src_key) or (ast_to_sym_id.get(src_key) if is_ast_id else None)
                        tested_name = target_name
                    else:
                        tested_name = src_key

                    if test_id:
                        tested_id = qname_to_sym_id.get(tested_name)
                        if not tested_id:
                            targets = self.db.find_symbols(tested_name, limit=5)
                            exact = [t for t in targets if t['name'] == tested_name and t['id'] != test_id]
                            if exact:
                                tested_id = exact[0]['id']
                            elif targets:
                                valid = [t for t in targets if t['id'] != test_id]
                                if valid:
                                    tested_id = valid[0]['id']

                        if tested_id and tested_id != test_id:
                            edge_records.append((tested_id, test_id, 'TESTED_BY'))
                        else:
                            unresolved_records.append((test_id, tested_name, 'TESTED_BY'))
                    continue

                if is_ast_id:
                    source_id = ast_to_sym_id.get(src_key)
                else:
                    source_id = qname_to_sym_id.get(src_key)
                if not source_id:
                    continue

                target_id = qname_to_sym_id.get(target_name)
                if not target_id:
                    targets = self.db.find_symbols(target_name, limit=1)
                    target_id = targets[0]['id'] if targets else None

                if target_id:
                    edge_records.append((source_id, target_id, kind))
                else:
                    unresolved_records.append((source_id, target_name, kind))

            self.db.add_symbol_edges_batch(edge_records)
            self.db.add_unresolved_refs_batch(unresolved_records)

        return ReindexStatus.QUEUED_FOR_EMBEDDING if embed_added else ReindexStatus.PROCESSED_NO_EMBEDDING

    def _reindex_file(self, file_id: int, filepath: Path, rel_path: str, chunks_to_embed: list) -> ReindexStatus:
        """
        Совместимый метод (используется при последовательной обработке).
        Очищает старые данные файла, парсит и флашит результаты в БД.
        """
        self.db.clear_file_chunks(file_id)
        self.db.clear_file_symbols(file_id)

        parsed = self._parse_file_pure(filepath, rel_path)
        if parsed is None:
            return ReindexStatus.PARSE_ERROR
        return self._flush_parsed_file(file_id, rel_path, parsed, chunks_to_embed)


    # ──────────────────────────────────────────────────────────────────────
    # Эмбеддинги и отчётность
    # ──────────────────────────────────────────────────────────────────────

    def _embed_pending_chunks(self, repo_id: str, chunks_to_embed: list, sync_buffer: dict):
        """
        Векторизует накопленные чанки порциями (sub-batch) для экономии RAM.
        Дополнительно подхватывает чанки без эмбеддингов из предыдущих запусков (recovery после OOM).
        Обновляет sync_buffer по завершении каждой порции.
        """
        # Recovery: чанки, которые есть в БД, но потеряли вектор из-за OOM
        missing = self.db.get_chunks_without_embeddings(repo_id)
        if missing:
            logger.info(f"Recovery: Found {len(missing)} text chunks in DB missing vector embeddings. Queuing them now.")
            chunks_to_embed.extend(missing)

        if not chunks_to_embed:
            return

        # Удаляем дубликаты по rowid (могут возникнуть при наложении новых чанков и recovery)
        unique_list = []
        seen_rowids = set()
        for item in chunks_to_embed:
            rowid = item[1]
            if rowid not in seen_rowids:
                unique_list.append(item)
                seen_rowids.add(rowid)
        
        chunks_to_embed = unique_list
        total_chunks = len(chunks_to_embed)
        logger.info(f"Batch computing local embeddings for {total_chunks} chunks (sub-batches of {EMBED_SUB_BATCH_SIZE})...")

        processed_chunks = 0
        for batch_start in range(0, total_chunks, EMBED_SUB_BATCH_SIZE):
            batch_slice = chunks_to_embed[batch_start:batch_start + EMBED_SUB_BATCH_SIZE]
            texts = [item[0] for item in batch_slice]
            rowids = [item[1] for item in batch_slice]
            rp_list = [item[2] for item in batch_slice]

            vectors = self.embedder.embed_batch(texts, batch_size=8)

            processed_chunks += len(batch_slice)
            logger.info(f"Embedding progress: {processed_chunks}/{total_chunks} chunks ({int(processed_chunks/total_chunks*100)}%)")

            records = []
            completed_rp = set()
            for i, vector in enumerate(vectors):
                if vector:
                    records.append((rowids[i], vector))
                    completed_rp.add(rp_list[i])

            if records:
                self.db.add_embeddings_batch(records)

            gc.collect()

            # Обновляем sync_buffer для файлов текущей порции
            for rp in rp_list:
                if rp in sync_buffer:
                    sync_buffer[rp] = 'completed' if rp in completed_rp else 'error_embedding'

            # Освобождаем ссылки на тексты текущей порции
            del texts, vectors, records

        # Освобождаем весь буфер после завершения
        chunks_to_embed.clear()

    def _write_sync_report(self, repo_id: str, sync_buffer: dict):
        """Записывает JSON-отчёт о проблемных файлах или удаляет старый при 100% успехе."""
        # Санитизируем repo_id для безопасного имени файла
        safe_repo_id = repo_id.replace('/', '_').replace('\\', '_').replace('..', '_')
        pending_or_error = {k: v for k, v in sync_buffer.items() if v != 'completed'}
        report_path = self.db.db_path.parent / f"sync_report_{safe_repo_id}.json"

        if not pending_or_error:
            logger.info(f"Verification successful: 100% of {len(sync_buffer)} files processed correctly.")
            if report_path.exists():
                report_path.unlink()
        else:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "total": len(sync_buffer),
                    "successful": len(sync_buffer) - len(pending_or_error),
                    "issues": pending_or_error,
                }, f, indent=2)
            logger.error(f"Verification failed: {len(pending_or_error)} items failed. Report dumped to {report_path}")

    # ──────────────────────────────────────────────────────────────────────
    # Публичный API
    # ──────────────────────────────────────────────────────────────────────

    def index_file(
        self,
        file_path: Path | str,
        repo_id: str,
        project_id: str = "default",
        branch: str = "master",
        rel_path: Optional[str] = None,
        repo_root: Optional[Path | str] = None,
        force: bool = False
    ) -> int:
        """
        Индексирует один файл с поддержкой CAS-дедупликации, связей TESTED_BY и FTS5.
        Если файл с таким content_hash уже проиндексирован (например, в другой ветке/ворктри),
        пропускает повторное построение chunks, embeddings и symbols, привязывая существующий
        content_hash к новой записи в files — это сердцевина CAS-дедупликации!
        """
        file_path = Path(file_path).resolve()
        if not file_path.is_file():
            logger.warning(f"File does not exist: {file_path}")
            return 0

        if rel_path is None:
            if repo_root is not None:
                try:
                    rel_path = file_path.relative_to(Path(repo_root).resolve()).as_posix()
                except ValueError:
                    rel_path = file_path.name
            else:
                rel_path = file_path.name
        rel_path = rel_path.replace('\\', '/')

        try:
            content_bytes = file_path.read_bytes()
        except Exception as e:
            logger.error(f"Cannot read file {file_path}: {e}")
            return 0

        content_hash = hashlib.sha256(content_bytes).hexdigest()
        content_str = content_bytes.decode('utf-8', errors='replace')
        mtime = file_path.stat().st_mtime
        language = self._detect_language(file_path)

        # 1. Сохраняем сырой контент в CAS таблицу file_contents
        self.db.upsert_file_content(content_str, language=language)

        # 2. Проверяем наличие chunks для данного content_hash в БД (CAS-дедупликация)
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT 1 FROM chunks WHERE content_hash = ? LIMIT 1", (content_hash,))
        content_already_indexed = cursor.fetchone() is not None

        # 3. Сохраняем/обновляем запись в files
        file_id = self.db.upsert_file(
            repo_id=repo_id,
            path=rel_path,
            mtime=mtime,
            file_hash=content_hash,
            project_id=project_id,
            branch=branch,
            content_hash=content_hash
        )

        # 4. Если контент уже был проиндексирован, пропускаем перепарсинг chunks/symbols/embeddings
        if content_already_indexed and not force:
            logger.info(f"CAS hit: file '{rel_path}' ({content_hash[:8]}) already indexed. Deduplicating.")
            return file_id

        # 5. Иначе парсим файл, создаем чанки, символы и рёбра
        self.db.clear_file_chunks(file_id)
        self.db.clear_file_symbols(file_id)

        parsed = self._parse_file_pure(file_path, rel_path)
        if parsed is not None:
            chunks_to_embed = []
            self._flush_parsed_file(file_id, rel_path, parsed, chunks_to_embed, content_hash=content_hash)
            if self.use_embeddings and self.embedder and chunks_to_embed:
                self._embed_pending_chunks(repo_id, chunks_to_embed, {rel_path: 'pending'})

        return file_id

    def sync_repo(
        self,
        repo_id: str,
        repo_path: str | Path,
        allowed_top_level: list[str] | None = None,
        force: bool = False,
        target_files: list[str] | None = None,
        project_id: str = "default",
        branch: str = "master"
    ):
        """
        Инкрементальная синхронизация репозитория (дельта-скан).
        Сравнивает mtime и hash, удаляет старые файлы, парсит новые/измененные.
        Хэширование выполняется параллельно, а парсинг AST — в главном потоке.
        При force=True проверки mtime/hash игнорируются и все файлы переиндексируются.
        При передаче target_files проверяются и обновляются только указанные файлы (дельта вебхука).
        """
        repo_path = Path(repo_path).resolve()
        logger.info(f"Starting sync for repo '{repo_id}' (project='{project_id}', branch='{branch}') at {repo_path} (force={force}, target_files={len(target_files) if target_files else 'all'})")

        from knowledge_mcp.daemon.status_ws import broadcast_sync
        broadcast_sync({
            "type": "sync_progress",
            "data": {
                "repo_id": repo_id,
                "project_id": project_id,
                "branch": branch,
                "progress": 5,
                "status": "scanning"
            }
        })

        if not repo_path.exists():
            logger.error(f"Path does not exist: {repo_path}")
            return

        spec = self._get_ignore_spec(repo_path)

        # Семантический анализ .NET проектов через Roslyn
        has_csharp = (
            any(f.endswith('.cs') or f.endswith('.csproj') for f in target_files)
            if target_files is not None
            else True
        )
        if has_csharp:
            self._run_roslyn_analysis(repo_path, spec)

        # Конвертируем sqlite3.Row в чистые dict для thread-safety
        known_files_raw = self.db.get_known_files(repo_id, project_id=project_id, branch=branch)
        known_files = {path: dict(row) for path, row in known_files_raw.items()}

        current_files = set()
        updated_count = 0
        added_count = 0
        unchanged_count = 0
        deleted_count = 0

        files_to_check = []
        if target_files is not None:
            logger.info(f"Targeted delta sync for {len(target_files)} files in repo '{repo_id}'")
            norm_target_files = {tf.replace('\\', '/').lstrip('/') for tf in target_files if tf.strip()}
            for rel_path in norm_target_files:
                filepath = repo_path / rel_path
                if filepath.is_file():
                    if spec and spec.match_file(rel_path):
                        continue
                    files_to_check.append((filepath, rel_path))
                    current_files.add(rel_path)
                elif rel_path in known_files:
                    # Файл удален из репозитория
                    self.db.delete_file(known_files[rel_path]['id'])
                    deleted_count += 1
        else:
            logger.info("Walking directory tree to collect files...")
            for filepath in self._walk_files(repo_path, spec, allowed_top_level):
                if filepath.is_file():
                    rel_path = filepath.relative_to(repo_path).as_posix()
                    files_to_check.append((filepath, rel_path))
                    current_files.add(rel_path)

        def check_file(item):
            """
            Параллельная задача: хэширует файл. Не делает никаких обращений к БД.
            Возвращает: (rel_path, filepath, mtime, file_hash, status, old_id)
            """
            filepath, rel_path = item
            try:
                mtime = filepath.stat().st_mtime
                if not force and rel_path in known_files:
                    known = known_files[rel_path]
                    is_cs = filepath.suffix.lower() == '.cs'
                    # Защита от пропуска C# файлов после сбоя парсера:
                    # если в БД нет записей в symbols или chunks для непустого .cs файла, форсируем повторный парсинг ('updated')
                    is_orphan = (
                        is_cs
                        and filepath.stat().st_size > 0
                        and (known.get('chunks_count', 0) == 0 or known.get('symbols_count', 0) == 0)
                    )

                    if not is_orphan:
                        if known['mtime'] == mtime:
                            return (rel_path, filepath, mtime, None, 'unchanged', None)
                        file_hash = hash_file(filepath)
                        if known['hash'] == file_hash:
                            return (rel_path, filepath, mtime, file_hash, 'touch', None)
                        return (rel_path, filepath, mtime, file_hash, 'updated', known['id'])

                # Если force, или файл новый, или файл-сирота, или изменился:
                file_hash = hash_file(filepath)
                status = 'updated' if rel_path in known_files else 'added'
                old_id = known_files[rel_path]['id'] if rel_path in known_files else None
                return (rel_path, filepath, mtime, file_hash, status, old_id)
            except Exception as e:
                return (rel_path, filepath, None, None, 'error', str(e))

        sync_buffer = {rel_path: 'pending' for _, rel_path in files_to_check}
        chunks_to_embed = []

        # Оборачиваем весь цикл в одну транзакцию вместо тысяч отдельных commit()
        self.db.begin_transaction()
        try:
            # Хэширование файлов параллельно
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_HASH_WORKERS) as executor:
                total_files = len(files_to_check)
                logger.info(f"Found {total_files} files to check. Starting content hashing and AST parsing...")
                results = executor.map(check_file, files_to_check)

                processed = 0
                log_step = max(1, total_files // 10) if total_files > 0 else 1

                for res in results:
                    processed += 1
                    if processed % log_step == 0 or processed == total_files:
                        logger.info(f"Scan progress: {processed}/{total_files} files ({int(processed/total_files*100)}%)")
                        current_progress = 10 + int(processed / total_files * 70)
                        from knowledge_mcp.daemon.status_ws import broadcast_sync
                        broadcast_sync({
                            "type": "sync_progress",
                            "data": {
                                "repo_id": repo_id,
                                "progress": current_progress,
                                "status": "syncing"
                            }
                        })

                    rel_path, filepath, mtime, file_hash, status, old_id = res
                    if status == 'error':
                        logger.error(f"Error processing {rel_path}: {old_id}")
                        sync_buffer[rel_path] = 'error'
                    elif status == 'unchanged':
                        unchanged_count += 1
                        sync_buffer[rel_path] = 'completed'
                    elif status == 'touch':
                        self.db.upsert_file(repo_id, rel_path, mtime, file_hash, project_id=project_id, branch=branch)
                        unchanged_count += 1
                        sync_buffer[rel_path] = 'completed'
                    elif status in ('updated', 'added'):
                        if status == 'updated':
                            updated_count += 1
                        else:
                            added_count += 1

                        # Сохраняем сырой контент в CAS таблицу file_contents
                        try:
                            content_str = filepath.read_text(encoding='utf-8', errors='replace')
                        except Exception:
                            content_str = ""
                        self.db.upsert_file_content(content_str, language=self._detect_language(filepath))

                        # Проверяем CAS-кэш: есть ли уже чанки для такого file_hash
                        cursor = self.db.conn.cursor()
                        cursor.execute("SELECT 1 FROM chunks WHERE content_hash = ? LIMIT 1", (file_hash,))
                        has_chunks = cursor.fetchone() is not None

                        file_id = self.db.upsert_file(repo_id, rel_path, mtime, file_hash, project_id=project_id, branch=branch, content_hash=file_hash)

                        if has_chunks and not force:
                            sync_buffer[rel_path] = 'completed'
                        else:
                            self.db.clear_file_chunks(file_id)
                            self.db.clear_file_symbols(file_id)

                            # Парсинг AST в главном потоке
                            parsed = self._parse_file_pure(filepath, rel_path)

                            if parsed is None:
                                sync_buffer[rel_path] = 'skipped_parse_error'
                            else:
                                reindex_result = self._flush_parsed_file(
                                    file_id, rel_path, parsed, chunks_to_embed, content_hash=file_hash
                                )
                                if reindex_result == ReindexStatus.QUEUED_FOR_EMBEDDING:
                                    sync_buffer[rel_path] = 'waiting_for_embedding'
                                else:
                                    sync_buffer[rel_path] = 'completed'

                    # Коммитим батчами каждые BATCH_COMMIT_SIZE файлов, чтобы не держать
                    # гигантскую транзакцию в памяти и не вызывать disk I/O error в SQLite
                    if processed % BATCH_COMMIT_SIZE == 0:
                        self.db.commit_transaction()
                        self.db.begin_transaction()

            # Удаляем из БД ушедшие с диска файлы (только при полном обходе репозитория)
            if target_files is None:
                for rel_path, known in known_files.items():
                    if rel_path not in current_files:
                        self.db.delete_file(known['id'])
                        deleted_count += 1

            self.db.commit_transaction()
        except Exception as e:
            self.db.rollback_transaction()
            logger.error(f"Transaction failed during sync of '{repo_id}': {e}")
            raise

        # Очищаем тяжелый кэш семантики перед самой ресурсоемкой фазой векторизации
        self.csharp_cache = {}
        gc.collect()

        # Батч-векторизация + recovery потерянных эмбеддингов
        if self.use_embeddings and self.embedder:
            from knowledge_mcp.daemon.status_ws import broadcast_sync
            broadcast_sync({
                "type": "sync_progress",
                "data": {
                    "repo_id": repo_id,
                    "progress": 80,
                    "status": "embedding"
                }
            })
            self._embed_pending_chunks(repo_id, chunks_to_embed, sync_buffer)

        # Финальная верификация: отчёт о проблемных файлах
        self._write_sync_report(repo_id, sync_buffer)

        # Разрешение повисших cross-repo ссылок
        try:
            resolved_count = self.db.resolve_pending_references()
            if resolved_count > 0:
                logger.info(f"Resolved {resolved_count} pending cross-repo references.")
        except Exception as e:
            logger.error(f"Error resolving pending references: {e}")

        from knowledge_mcp.daemon.status_ws import broadcast_sync
        broadcast_sync({
            "type": "sync_progress",
            "data": {
                "repo_id": repo_id,
                "progress": 100,
                "status": "completed"
            }
        })
        logger.info(f"Sync complete for '{repo_id}': +{added_count} ~{updated_count} -{deleted_count} (={unchanged_count} unchanged)")
        try:
            self.db.optimize()
        except Exception as e:
            logger.warning(f"Failed to optimize database after sync: {e}")

    def sync_project(self, project_config: dict, force: bool = False) -> dict:
        """Синхронизирует мульти-репозиторный воркспейс со всеми репозиториями и документацией."""
        from knowledge_mcp.core.workspace_scanner import WorkspaceScanner
        scanner = WorkspaceScanner()
        targets = scanner.scan_workspace(project_config)
        logger.info(f"Syncing project '{project_config.get('id')}' with {len(targets)} targets...")

        results = {"total_targets": len(targets), "repos_synced": 0, "docs_synced": 0, "errors": 0}
        for target in targets:
            try:
                if target.is_doc:
                    self.index_file(
                        target.absolute_path,
                        repo_id=target.repo_id,
                        project_id=target.project_id,
                        branch=target.branch,
                        force=force
                    )
                    results["docs_synced"] += 1
                else:
                    self.sync_repo(
                        repo_id=target.repo_id,
                        repo_path=target.absolute_path,
                        project_id=target.project_id,
                        branch=target.branch,
                        force=force
                    )
                    results["repos_synced"] += 1
            except Exception as e:
                logger.error(f"Failed syncing target {target.absolute_path}: {e}")
                results["errors"] += 1

        return results

