import logging
import sys
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

# ─── MAIN PIPELINE ──────────────────────────────────────────────
def main():
    start_time = datetime.now()
    log.info("=" * 50)
    log.info("SIDRA'S BRIEFING PIPELINE STARTING")
    log.info("=" * 50)

    steps = [
        ("Ingestion", run_ingest),
        ("Ranking", run_ranker),
        ("Summarization", run_summarizer),
        ("Delivery", run_delivery),
    ]

    for step_name, step_func in steps:
        try:
            log.info(f"── Starting {step_name}...")
            step_func()
            log.info(f"✓ {step_name} complete")
        except Exception as e:
            log.error(f"✗ {step_name} failed: {e}")
            log.error("Pipeline stopped due to error")
            sys.exit(1)

    elapsed = (datetime.now() - start_time).total_seconds()
    log.info("=" * 50)
    log.info(f"PIPELINE COMPLETE in {elapsed:.2f} seconds")
    log.info("=" * 50)

if __name__ == "__main__":
    main()