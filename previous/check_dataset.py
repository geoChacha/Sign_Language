import sqlite3, json
from pathlib import Path

conn = sqlite3.connect('main_dataset.db')
c = conn.cursor()

c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in c.fetchall()]

for t in tables:
    c.execute(f"SELECT COUNT(*) FROM {t}")
    n = c.fetchone()[0]
    c.execute(f"SELECT DISTINCT label FROM {t}")
    labels = sorted([r[0] for r in c.fetchall()])
    c.execute(f"SELECT label, COUNT(*) FROM {t} GROUP BY label ORDER BY COUNT(*) DESC")
    counts = c.fetchall()
    print(f"\n{'='*50}")
    print(f"Table: {t}  ({n} rows, {len(labels)} classes)")
    for lbl, cnt in counts:
        print(f"  {lbl:<20} {cnt:4d} samples")

conn.close()

# Check PSL_dataset folder structure
print(f"\n{'='*50}")
print("PSL_dataset folder structure:")
for d in ['PSL_dataset/datasets/words_dataset', 'PSL_dataset/datasets/alphabets_dataset']:
    base = Path(d)
    if base.exists():
        folders = sorted([f for f in base.iterdir() if f.is_dir()])
        total = sum(len(list(f.rglob('*.json'))) for f in folders)
        print(f"\n{d}: {len(folders)} classes, {total} total JSON files")
        for f in folders:
            n = len(list(f.rglob('*.json')))
            print(f"  {f.name[:30]:<30} {n:4d} files")
