import logging
import sys
import json
import sqlite3
from datetime import datetime

# ─── LOGGING ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── IMPORT PIPELINE STAGES ─────────────────────────────────────
from ingest import main as run_ingest
from ranker import main as run_ranker
from summarizer import main as run_summarizer
from delivery import main as run_delivery
from metrics import record_run, generate_dashboard
from init_db import initialize_database

# ─── COLLECT METRICS HELPERS ────────────────────────────────────
def count_articles_by_origin():
    try:
        with open("raw_news.json", "r") as f:
            articles = json.load(f)
        newsapi = sum(1 for a in articles if a.get("origin") == "newsapi")
        guardian = sum(1 for a in articles if a.get("origin") == "guardian")
        rss = sum(1 for a in articles if a.get("origin") == "rss")
        return len(articles), newsapi, guardian, rss
    except Exception:
        return 0, 0, 0, 0

def count_priority_distribution():
    try:
        conn = sqlite3.connect("pipeline.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT priority, COUNT(*) FROM articles
            WHERE inserted_at >= date('now', '-1 day')
            GROUP BY priority
        """)
        rows = dict(cursor.fetchall())
        conn.close()
        return rows.get("High", 0), rows.get("Medium", 0), rows.get("Low", 0)
    except Exception:
        return 0, 0, 0

def get_briefing_length():
    try:
        with open("briefing.txt", "r", encoding="utf-8") as f:
            return len(f.read())
    except FileNotFoundError:
        return 0

def get_classifier_accuracy():
    """Read latest accuracy from model_comparison.json if available."""
    try:
        with open("model_comparison.json", "r") as f:
            data = json.load(f)
        return data.get("best_accuracy", 0.0)
    except FileNotFoundError:
        return 0.0

# ─── MAIN PIPELINE ──────────────────────────────────────────────
def main():
    start_time = datetime.now()
    log.info("=" * 50)
    log.info("SIDRA'S BRIEFING PIPELINE STARTING")
    log.info("=" * 50)

    # Ensure DB exists
    initialize_database()

    steps = [
        ("Ingestion", run_ingest),
        ("Ranking", run_ranker),
        ("Summarization", run_summarizer),
        ("Delivery", run_delivery),
    ]

    briefing_generated = False
    for step_name, step_func in steps:
        try:
            log.info(f"── Starting {step_name}...")
            step_func()
            log.info(f"✓ {step_name} complete")
            if step_name == "Summarization":
                briefing_generated = get_briefing_length() > 0
        except Exception as e:
            log.error(f"✗ {step_name} failed: {e}")
            log.error("Pipeline stopped due to error")
            sys.exit(1)

    elapsed = (datetime.now() - start_time).total_seconds()

    # ─── RECORD METRICS ─────────────────────────────────────────
    total_articles, newsapi, guardian, rss = count_articles_by_origin()
    high, medium, low = count_priority_distribution()

    # Estimate duplicates blocked (raw count minus deduped count)
    try:
        with open("raw_news.json", "r") as f:
            deduped_count = len(json.load(f))
    except Exception:
        deduped_count = total_articles

    record_run(
        articles_ingested=total_articles,
        articles_after_dedup=deduped_count,
        duplicates_blocked=max(0, total_articles - deduped_count),
        high_priority_count=high,
        medium_priority_count=medium,
        low_priority_count=low,
        sources_newsapi=newsapi,
        sources_guardian=guardian,
        sources_rss=rss,
        classifier_accuracy=get_classifier_accuracy(),
        pipeline_runtime_seconds=elapsed,
        briefing_length=get_briefing_length(),
        briefing_generated=briefing_generated
    )

    # ─── GENERATE DASHBOARD ─────────────────────────────────────
    generate_dashboard()

    log.info("=" * 50)
    log.info(f"PIPELINE COMPLETE in {elapsed:.2f} seconds")
    log.info("=" * 50)

if __name__ == "__main__":
    main()