import os
import sqlite3

db_path = 'data/knowledge.db'
if not os.path.exists(db_path):
    print('DB not found at', db_path)
    exit(1)

size_mb = os.path.getsize(db_path) / (1024 * 1024)
print(f"Database file size: {size_mb:.2f} MB")

conn = sqlite3.connect(db_path)
c = conn.cursor()

page_size = c.execute('PRAGMA page_size;').fetchone()[0]
page_count = c.execute('PRAGMA page_count;').fetchone()[0]
freelist_count = c.execute('PRAGMA freelist_count;').fetchone()[0]
journal_mode = c.execute('PRAGMA journal_mode;').fetchone()[0]

print(f"Page size: {page_size} bytes")
print(f"Total pages: {page_count} (Allocated: {(page_count*page_size)/(1024*1024):.2f} MB)")
print(f"Freelist pages: {freelist_count} ({(freelist_count*page_size)/(1024*1024):.2f} MB)")
print(f"Journal mode: {journal_mode}")

tables = [row[0] for row in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
print("\nTables and row counts:")
for t in tables:
    try:
        cnt = c.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
        print(f"  - {t:30}: {cnt:8} rows")
    except Exception as e:
        print(f"  - {t:30}: error ({e})")

# dbstat если доступен
try:
    print("\nDetailed space distribution (dbstat):")
    stat = c.execute("""
        SELECT name, sum(pgsize) as total_bytes, count(*) as pages
        FROM dbstat
        GROUP BY name
        ORDER BY total_bytes DESC
        LIMIT 15;
    """).fetchall()
    for name, bytes_used, pgs in stat:
        print(f"  - {name:30}: {bytes_used / 1024:.1f} KB ({pgs} pages)")
except Exception:
    print("\ndbstat virtual table not available in standard sqlite3 build.")

conn.close()
