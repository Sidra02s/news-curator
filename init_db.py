import sqlite3

def initialize_database():
    connection = sqlite3.connect("pipeline.db")
    cursor = connection.cursor()

    # ─── ARTICLES TABLE ──────────────────────────────────────────
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

    # ─── PIPELINE RUNS TABLE ─────────────────────────────────────
    # Tracks per-run metrics for the dashboard
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            articles_ingested INTEGER,
            articles_after_dedup INTEGER,
            duplicates_blocked INTEGER,
            high_priority_count INTEGER,
            medium_priority_count INTEGER,
            low_priority_count INTEGER,
            sources_newsapi INTEGER,
            sources_guardian INTEGER,
            sources_rss INTEGER,
            classifier_accuracy REAL,
            pipeline_runtime_seconds REAL,
            briefing_length INTEGER,
            briefing_generated INTEGER
        )
    """)

    connection.commit()
    connection.close()
    print("🚀 Database initialized successfully! pipeline.db created.")

if __name__ == "__main__":
    initialize_database()