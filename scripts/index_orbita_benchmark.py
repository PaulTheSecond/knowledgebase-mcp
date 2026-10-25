import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_mcp.db.models import KnowledgeDB
from knowledge_mcp.core.indexer import Indexer
from knowledge_mcp.core.config import load_config, add_or_update_project

def main():
    db_path = Path("data/knowledge.db")
    print(f"==================================================")
    print(f"  ORBITA WORKSPACE INDEXATION BENCHMARK")
    print(f"==================================================")
    print(f"Target Database: {db_path.resolve()}")
    
    db_size_before = db_path.stat().st_size if db_path.exists() else 0
    print(f"DB Size Before: {db_size_before / (1024*1024):.2f} MB")
    
    project_config = {
        "id": "orbita",
        "name": "Orbita ATS Workspace",
        "workspace_root": "C:/Repos/DVCN/Orbita",
        "type": "worktree_workspace",
        "base_path": "worktrees/master",
        "tasks_path": "worktrees/{task}",
        "docs_paths": [
            ".kb",
            "CLAUDE.md",
            "work-orders",
            "test-results",
            "ONBOARDING.md",
            "AGENTS.md"
        ],
        "repositories": [
            {"id": "backend", "path": "backend", "language": "csharp"},
            {"id": "frontend", "path": "frontend", "language": "typescript"},
            {"id": "smartapp-backend", "path": "smartapp-backend", "language": "csharp"},
            {"id": "smartapp-frontend", "path": "smartapp-frontend", "language": "typescript"},
            {"id": "browser-extension", "path": "browser-extension", "language": "typescript"},
            {"id": "e2e-tests", "path": "e2e-tests", "language": "typescript"},
            {"id": "devops", "path": "devops", "language": "yaml"}
        ]
    }
    
    # Save project to config.yaml
    add_or_update_project(project_config)
    print("Added project 'orbita' to data/config.yaml")
    
    db = KnowledgeDB(db_path)
    cfg = load_config()
    use_embeddings = cfg.get("embeddings", {}).get("enabled", False)
    print(f"Neural Embeddings enabled: {use_embeddings}")
    
    indexer = Indexer(db=db, use_embeddings=use_embeddings)
    
    print("\nStarting indexing of Orbita workspace...")
    t_start = time.perf_counter()
    
    results = indexer.sync_project(project_config)
    
    t_end = time.perf_counter()
    total_seconds = t_end - t_start
    
    db_size_after = db_path.stat().st_size
    size_diff_mb = (db_size_after - db_size_before) / (1024 * 1024)
    
    cas_stats = db.get_cas_stats(project_id="orbita")
    
    # Symbols count
    cur = db.conn.cursor()
    cur.execute("SELECT COUNT(*) FROM symbols s JOIN files f ON s.file_id = f.id WHERE f.project_id = 'orbita'")
    orbita_symbols = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM chunks c JOIN files f ON c.file_id = f.id WHERE f.project_id = 'orbita'")
    orbita_chunks = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM files WHERE project_id = 'orbita'")
    orbita_files = cur.fetchone()[0]
    
    print(f"\n==================================================")
    print(f"  BENCHMARK RESULTS")
    print(f"==================================================")
    print(f"⏱️  Total Time: {total_seconds:.2f} seconds ({total_seconds/60:.2f} minutes)")
    print(f"📁 Targets Synced: {results.get('repos_synced', 0)} repositories, {results.get('docs_synced', 0)} doc targets")
    print(f"📄 Files Indexed in DB: {orbita_files}")
    print(f"🧩 Chunks Created: {orbita_chunks}")
    print(f"🔍 Symbols Extracted: {orbita_symbols}")
    print(f"💾 DB Size Before: {db_size_before / (1024*1024):.2f} MB")
    print(f"💾 DB Size After:  {db_size_after / (1024*1024):.2f} MB (+{size_diff_mb:.2f} MB)")
    print(f"📊 CAS Stats for Orbita: {cas_stats}")
    print(f"==================================================")

if __name__ == "__main__":
    main()
