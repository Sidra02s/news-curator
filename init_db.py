import sqlite3

def initialize_database():
    # Connect to SQLite (this automatically creates pipeline.db if it doesn't exist)
    connection = sqlite3.connect("pipeline.db")
    cursor = connection.cursor()

    # Create the articles table blueprint
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            url TEXT UNIQUE NOT NULL,
            source TEXT,
            priority TEXT,
            summary TEXT,
            published_at TEXT,
            inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Save changes and close the connection
    connection.commit()
    connection.close()
    print("🚀 Database initialized successfully! pipeline.db created.")

if __name__ == "__main__":
    initialize_database()