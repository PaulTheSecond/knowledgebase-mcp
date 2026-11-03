"""Репозиторий symbols (инфраструктурный слой доступа к данным)."""

import hashlib
import json
import logging
import sqlite3
import struct
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import sqlite_vec
except ImportError:
    sqlite_vec = None

RRF_K = 60

from knowledge_mcp.infrastructure.db.repositories.base import BaseSqliteRepository

logger = logging.getLogger("knowledge_mcp.db.symbols")


class SqliteSymbolRepository(BaseSqliteRepository):
    """Специализированный репозиторий SQLite для symbols."""

    def get_impact_analysis(
        self,
        symbol_id: int,
        max_depth: int = 3,
        project_id: Optional[str] = None,
        branch: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        
        visited = set([symbol_id])
        queue: deque = deque([(symbol_id, 1)])
        impacted = []

        # Предварительный шаг: связываем интерфейс и реализацию для корневого символа
        # 1. Если символ реализует интерфейс, добавляем интерфейс и методы интерфейса
        cursor.execute("SELECT target_id FROM symbol_edges WHERE source_id = ? AND kind = 'IMPLEMENTS'", (symbol_id,))
        for (iface_id,) in cursor.fetchall():
            if iface_id not in visited:
                visited.add(iface_id)
                queue.append((iface_id, 1))
            cursor.execute("SELECT id FROM symbols WHERE file_id = (SELECT file_id FROM symbols WHERE id = ?)", (iface_id,))
            for (m_id,) in cursor.fetchall():
                if m_id not in visited:
                    visited.add(m_id)
                    queue.append((m_id, 1))

        # 2. Если символ это интерфейс, добавляем классы-реализации
        cursor.execute("SELECT source_id FROM symbol_edges WHERE target_id = ? AND kind = 'IMPLEMENTS'", (symbol_id,))
        for (impl_id,) in cursor.fetchall():
            if impl_id not in visited:
                visited.add(impl_id)
                queue.append((impl_id, 1))

        # 3. Если символ это метод реализации, находим соответствующий метод интерфейса
        cursor.execute("""
            SELECT s2.id 
            FROM symbols s1
            JOIN symbol_edges e ON e.source_id = (
                SELECT s_p.id FROM symbols s_p 
                WHERE s_p.file_id = s1.file_id AND s_p.kind IN ('class', 'struct') 
                LIMIT 1
            )
            JOIN symbols s2 ON s2.file_id = (SELECT file_id FROM symbols WHERE id = e.target_id) AND s2.name = s1.name
            WHERE s1.id = ? AND e.kind = 'IMPLEMENTS'
        """, (symbol_id,))
        for (iface_m_id,) in cursor.fetchall():
            if iface_m_id not in visited:
                visited.add(iface_m_id)
                queue.append((iface_m_id, 1))

        # 4. Находим поля/параметры внедрения зависимостей (DI) типа данного интерфейса/класса
        cursor.execute("SELECT name, file_id, kind FROM symbols WHERE id = ?", (symbol_id,))
        s_row = cursor.fetchone()
        if s_row and s_row[0]:
            sym_name = s_row[0]
            targets_to_check = [sym_name]
            clean_name = sym_name.lstrip('I') if sym_name.startswith('I') and len(sym_name) > 1 and sym_name[1].isupper() else sym_name
            if clean_name != sym_name:
                targets_to_check.append(clean_name)
            else:
                targets_to_check.append(f"I{sym_name}")

            # Если метод или свойство определены внутри класса (C# OOP), проверяем DI родительского класса
            if s_row[2] in ('method', 'property'):
                cursor.execute("SELECT name FROM symbols WHERE file_id = ? AND kind IN ('class', 'struct') LIMIT 1", (s_row[1],))
                p_row = cursor.fetchone()
                if p_row and p_row[0]:
                    cls_name = p_row[0]
                    cls_clean = cls_name.lstrip('I') if cls_name.startswith('I') and len(cls_name) > 1 and cls_name[1].isupper() else cls_name
                    targets_to_check.extend([cls_name, cls_clean, f"I{cls_clean}"])

            targets_to_check = list(dict.fromkeys(t for t in targets_to_check if len(t) > 2 and not t.startswith('_')))
            if targets_to_check:
                if s_row[2] in ('class', 'interface', 'struct'):
                    di_target_1 = targets_to_check[0]
                    di_target_2 = targets_to_check[1] if len(targets_to_check) > 1 else targets_to_check[0]
                    cursor.execute("""
                        SELECT DISTINCT s.id, s.file_id, s.name, s.kind
                        FROM symbols s
                        WHERE s.kind IN ('field', 'property', 'parameter') 
                          AND (s.signature = ? OR s.signature = ? OR s.name = ? OR s.name = ?)
                    """, (di_target_1, di_target_2, di_target_1, di_target_2))
                    for di_id, di_file_id, di_name, di_kind in cursor.fetchall():
                        if di_id not in visited:
                            visited.add(di_id)
                        cursor.execute("""
                            SELECT s_parent.id, s_parent.name, s_parent.qualified_name, s_parent.kind,
                                   f.repo_id, f.path, f.project_id, f.branch
                            FROM symbols s_parent
                            JOIN files f ON s_parent.file_id = f.id
                            WHERE s_parent.file_id = ? AND s_parent.kind IN ('class', 'struct')
                        """, (di_file_id,))
                        for pr in cursor.fetchall():
                            parent_id = pr['id']
                            if parent_id not in visited:
                                visited.add(parent_id)
                                impacted.append({
                                    'symbol': dict(pr),
                                    'depth': 1,
                                    'edge_kind': 'USES_DI'
                                })
                                queue.append((parent_id, 2))

                placeholders = ','.join(['?'] * len(targets_to_check))
                cursor.execute(f"""
                    SELECT DISTINCT s.id, s.name, s.qualified_name, s.kind,
                                    f.repo_id, f.path, f.project_id, f.branch
                    FROM unresolved_refs u
                    JOIN symbols s ON u.source_id = s.id
                    JOIN files f ON s.file_id = f.id
                    WHERE u.target_qualified_name IN ({placeholders})
                """, targets_to_check)
                for ur in cursor.fetchall():
                    u_id = ur['id']
                    if u_id not in visited:
                        visited.add(u_id)
                        impacted.append({
                            'symbol': dict(ur),
                            'depth': 1,
                            'edge_kind': 'CALLS_UNRESOLVED'
                        })
                        queue.append((u_id, 2))

        while queue:
            current_id, depth = queue.popleft()
            if depth > max_depth:
                continue
                
            sql = '''
                SELECT DISTINCT s.*, f.repo_id, f.path, f.project_id, f.branch, e.kind as edge_kind
                FROM symbol_edges e
                JOIN symbols s ON e.source_id = s.id
                JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
                WHERE e.target_id = ? AND e.kind IN ('CALLS', 'INHERITS', 'IMPLEMENTS', 'IMPORTS', 'USES')
            '''
            params: list[Any] = [current_id]
            if project_id is not None:
                sql += " AND f.project_id = ?"
                params.append(project_id)
            if branch is not None:
                sql += " AND f.branch = ?"
                params.append(branch)
            cursor.execute(sql, params)
            
            callers = cursor.fetchall()
            for r in callers:
                caller_id = r['id']
                if caller_id not in visited:
                    visited.add(caller_id)
                    impacted.append({
                        'symbol': r,
                        'depth': depth,
                        'edge_kind': r['edge_kind']
                    })
                    queue.append((caller_id, depth + 1))
                    
        return impacted


    def add_symbol_edge(self, source_id: int, target_id: int, kind: str):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO symbol_edges (source_id, target_id, kind)
            VALUES (?, ?, ?)
        """, (source_id, target_id, kind))
        self._auto_commit()


    def add_unresolved_ref(self, source_id: int, target_qualified_name: str, kind: str):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO unresolved_refs (source_id, target_qualified_name, kind)
            VALUES (?, ?, ?)
        """, (source_id, target_qualified_name, kind))
        self._auto_commit()


    def get_callers(self, symbol_id: int) -> List[sqlite3.Row]:
        """Обратно-совместимый вызов получения callers."""
        return self.get_symbol_callers(symbol_id, project_id=None, branch=None)


    def clear_file_symbols(self, file_id: int):
        """Удаляет все символы, привязанные к файлу (связи удалятся каскадно)."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM symbols WHERE file_id = ?", (file_id,))
        self._auto_commit()


    def add_symbol_edges_batch(self, records: list[tuple]):
        """Батчевая вставка рёбер. records: [(source_id, target_id, kind), ...]"""
        if not records:
            return
        cursor = self.conn.cursor()
        cursor.executemany("""
            INSERT INTO symbol_edges (source_id, target_id, kind)
            VALUES (?, ?, ?)
        """, records)
        self._auto_commit()


    def add_symbol(
        self,
        file_id: int,
        name: str,
        qualified_name: str,
        kind: str,
        language: str,
        line_start: int,
        line_end: int,
        signature: str,
        chunk_id: str,
        content_hash: Optional[str] = None
    ) -> int:
        cursor = self.conn.cursor()
        if content_hash is None and file_id:
            cursor.execute("SELECT content_hash FROM files WHERE id = ?", (file_id,))
            row = cursor.fetchone()
            if row and row['content_hash']:
                content_hash = row['content_hash']
        cursor.execute("""
            INSERT INTO symbols (file_id, name, qualified_name, kind, language, line_start, line_end, signature, chunk_id, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (file_id, name, qualified_name, kind, language, line_start, line_end, signature, chunk_id, content_hash))
        self._auto_commit()
        return cursor.lastrowid


    def get_callees(self, symbol_id: int) -> List[sqlite3.Row]:
        """Обратно-совместимый вызов получения callees."""
        return self.get_symbol_callees(symbol_id, project_id=None, branch=None)


    def get_symbol_callees(
        self,
        symbol_id: int,
        project_id: Optional[str] = "default",
        branch: Optional[str] = "master"
    ) -> List[sqlite3.Row]:
        """Возвращает вызываемые символы (callees) для symbol_id."""
        cursor = self.conn.cursor()
        sql = '''
            SELECT DISTINCT s.*, f.repo_id, f.path, f.project_id, f.branch, e.kind as edge_kind
            FROM symbol_edges e
            JOIN symbols s ON e.target_id = s.id
            JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
            WHERE e.source_id = ?
        '''
        params: list[Any] = [symbol_id]
        if project_id is not None:
            sql += " AND f.project_id = ?"
            params.append(project_id)
        if branch is not None:
            sql += " AND f.branch = ?"
            params.append(branch)
        cursor.execute(sql, params)
        return cursor.fetchall()


    def get_symbols(
        self,
        name_pattern: str = "%",
        kind: Optional[str] = None,
        repo_ids: Optional[List[str]] = None,
        project_id: Optional[str] = "default",
        branch: Optional[str] = "master",
        limit: int = 20
    ) -> List[sqlite3.Row]:
        """Поиск символов с фильтрацией по project_id и branch."""
        cursor = self.conn.cursor()
        sql = '''
            SELECT DISTINCT s.*, f.repo_id, f.path, f.project_id, f.branch
            FROM symbols s
            JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
            WHERE (s.name LIKE ? OR s.qualified_name LIKE ?)
        '''
        # Защита от переполнения шаблона LIKE в SQLite (OperationalError: LIKE or GLOB pattern too complex)
        clean_pattern = name_pattern.splitlines()[0][:100] if name_pattern else ""
        sql_pattern = clean_pattern.replace('*', '%')
        if '%' not in sql_pattern:
            sql_pattern = f"%{sql_pattern}%"
            
        params: list[Any] = [sql_pattern, sql_pattern]
        
        if kind:
            sql += " AND s.kind = ?"
            params.append(kind)
            
        if repo_ids:
            placeholders = ','.join('?' for _ in repo_ids)
            sql += f" AND f.repo_id IN ({placeholders})"
            params.extend(repo_ids)

        if project_id is not None:
            sql += " AND f.project_id = ?"
            params.append(project_id)

        if branch is not None:
            sql += " AND f.branch = ?"
            params.append(branch)
            
        raw_name = clean_pattern.replace('*', '').replace('%', '')
        sql += """
            ORDER BY 
                CASE 
                    WHEN s.name = ? THEN 0
                    WHEN s.name LIKE ? THEN 1
                    WHEN s.qualified_name = ? THEN 2
                    ELSE 3
                END,
                CASE 
                    WHEN s.kind IN ('class', 'function', 'component', 'method') THEN 0
                    WHEN s.kind IN ('interface', 'type') THEN 1
                    ELSE 2
                END,
                s.id ASC
            LIMIT ?
        """
        params.extend([raw_name, f"{raw_name}%", raw_name, limit])
        
        cursor.execute(sql, params)
        return cursor.fetchall()


    def find_symbols(
        self,
        name_pattern: str,
        kind: Optional[str] = None,
        repo_ids: Optional[List[str]] = None,
        limit: int = 20,
        project_id: Optional[str] = None,
        branch: Optional[str] = None
    ) -> List[sqlite3.Row]:
        """Обратно-совместимый метод поиска символов."""
        return self.get_symbols(
            name_pattern=name_pattern,
            kind=kind,
            repo_ids=repo_ids,
            project_id=project_id,
            branch=branch,
            limit=limit
        )


    def add_symbols_batch(self, records: list[tuple]) -> list[int]:
        """Батчевая вставка символов.
        records: [(file_id, name, qname, kind, lang, ls, le, sig, chunk_id), ...]
        или [(file_id, name, qname, kind, lang, ls, le, sig, chunk_id, content_hash), ...]
        Возвращает список rowid вставленных символов."""
        if not records:
            return []
        cursor = self.conn.cursor()
        rowids = []
        for record in records:
            if len(record) == 9:
                file_id, name, qname, kind, lang, ls, le, sig, chunk_id = record
                content_hash = None
                if file_id:
                    cursor.execute("SELECT content_hash FROM files WHERE id = ?", (file_id,))
                    row = cursor.fetchone()
                    if row and row['content_hash']:
                        content_hash = row['content_hash']
                cursor.execute("""
                    INSERT INTO symbols (file_id, name, qualified_name, kind, language, line_start, line_end, signature, chunk_id, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (file_id, name, qname, kind, lang, ls, le, sig, chunk_id, content_hash))
            elif len(record) == 10:
                cursor.execute("""
                    INSERT INTO symbols (file_id, name, qualified_name, kind, language, line_start, line_end, signature, chunk_id, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, record)
            rowids.append(cursor.lastrowid)
        self._auto_commit()
        return rowids


    def get_symbol_callers(
        self,
        symbol_id: int,
        project_id: Optional[str] = "default",
        branch: Optional[str] = "master"
    ) -> List[sqlite3.Row]:
        """Возвращает вызывающие символы (callers) для symbol_id."""
        cursor = self.conn.cursor()
        sql = '''
            SELECT DISTINCT s.*, f.repo_id, f.path, f.project_id, f.branch, e.kind as edge_kind
            FROM symbol_edges e
            JOIN symbols s ON e.source_id = s.id
            JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
            WHERE e.target_id = ?
        '''
        params: list[Any] = [symbol_id]
        if project_id is not None:
            sql += " AND f.project_id = ?"
            params.append(project_id)
        if branch is not None:
            sql += " AND f.branch = ?"
            params.append(branch)
        cursor.execute(sql, params)
        return cursor.fetchall()


    def resolve_pending_references(self, limit: int = 5000) -> int:
        """
        Пытается найти таргеты для ссылок из unresolved_refs через быстрый пакетный JOIN.
        Ограничен limit для защиты от зависания при большом числе системных/внешних ссылок.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT u.id, u.source_id, u.kind, s.id as target_id
            FROM unresolved_refs u
            JOIN symbols s ON (s.qualified_name = u.target_qualified_name OR s.name = u.target_qualified_name)
            LIMIT ?
        """, (limit,))
        matches = cursor.fetchall()
        if not matches:
            return 0
        
        edges_to_insert = []
        ids_to_delete = []
        for row in matches:
            ref_id = row['id']
            src_id = row['source_id']
            kind = row['kind']
            target_id = row['target_id']
            if kind == 'TESTED_BY':
                edges_to_insert.append((target_id, src_id, 'TESTED_BY'))
            else:
                edges_to_insert.append((src_id, target_id, kind))
            ids_to_delete.append(ref_id)
            
        if edges_to_insert:
            cursor.executemany(
                "INSERT OR IGNORE INTO symbol_edges (source_id, target_id, kind) VALUES (?, ?, ?)",
                edges_to_insert
            )
            
        step = 900
        for i in range(0, len(ids_to_delete), step):
            batch = ids_to_delete[i:i+step]
            placeholders = ','.join('?' for _ in batch)
            cursor.execute(f"DELETE FROM unresolved_refs WHERE id IN ({placeholders})", batch)
        self._auto_commit()
            
        return len(ids_to_delete)


    def get_hierarchy(
        self,
        symbol_id: int,
        project_id: Optional[str] = None,
        branch: Optional[str] = None
    ) -> List[sqlite3.Row]:
        cursor = self.conn.cursor()
        sql = '''
            SELECT DISTINCT s.*, f.repo_id, f.path, f.project_id, f.branch, e.kind as edge_kind
            FROM symbol_edges e
            JOIN symbols s ON e.target_id = s.id
            JOIN files f ON (s.file_id = f.id OR (s.content_hash IS NOT NULL AND s.content_hash = f.content_hash))
            WHERE e.source_id = ? AND e.kind IN ('INHERITS', 'IMPLEMENTS')
        '''
        params: list[Any] = [symbol_id]
        if project_id is not None:
            sql += " AND f.project_id = ?"
            params.append(project_id)
        if branch is not None:
            sql += " AND f.branch = ?"
            params.append(branch)
        cursor.execute(sql, params)
        return cursor.fetchall()


    def add_unresolved_refs_batch(self, records: list[tuple]):
        """Батчевая вставка unresolved ссылок. records: [(source_id, target_qname, kind), ...]"""
        if not records:
            return
        cursor = self.conn.cursor()
        cursor.executemany("""
            INSERT INTO unresolved_refs (source_id, target_qualified_name, kind)
            VALUES (?, ?, ?)
        """, records)
        self._auto_commit()


