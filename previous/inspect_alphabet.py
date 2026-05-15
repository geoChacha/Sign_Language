import sqlite3, numpy as np

conn = sqlite3.connect('main_dataset.db')
c = conn.cursor()
c.execute("SELECT * FROM rightHandDataset LIMIT 3")
rows = c.fetchall()
c.execute("PRAGMA table_info(rightHandDataset)")
cols = [col[1] for col in c.fetchall()]
conn.close()

print("Columns:", cols)
print()
for row in rows:
    label = row[cols.index('label')]
    feats = [row[cols.index(col)] for col in cols if col not in ('id','label')]
    arr = np.array(feats)
    xs = arr[0::2]
    ys = arr[1::2]
    print(f"Label: {label}")
    print(f"  x1,y1 (wrist): {xs[0]:.2f}, {ys[0]:.2f}")
    print(f"  x2,y2:         {xs[1]:.2f}, {ys[1]:.2f}")
    print(f"  x range: [{xs.min():.1f}, {xs.max():.1f}]")
    print(f"  y range: [{ys.min():.1f}, {ys.max():.1f}]")
    print()
