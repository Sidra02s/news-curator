import sqlite3
conn = sqlite3.connect('pipeline.db')
conn.execute('CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT UNIQUE, title TEXT, topic TEXT, category TEXT, vote TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)')
conn.commit()
conn.close()
print('done')