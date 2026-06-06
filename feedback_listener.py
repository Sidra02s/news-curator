import os
import json
import sqlite3
import requests
import logging
import time
from dotenv import load_dotenv

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

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

# ─── TELEGRAM HELPERS ───────────────────────────────────────────
def get_updates(offset=None):
    params = {"timeout": 30, "allowed_updates": ["callback_query"]}
    if offset:
        params["offset"] = offset
    try:
        r = requests.get(f"{BASE_URL}/getUpdates", params=params, timeout=35)
        return r.json()
    except Exception as e:
        log.error(f"Failed to get updates: {e}")
        return {"result": []}

def answer_callback(callback_id):
    """Clears the loading spinner on the button after it's tapped."""
    try:
        requests.post(f"{BASE_URL}/answerCallbackQuery",
                      json={"callback_query_id": callback_id}, timeout=10)
    except Exception as e:
        log.warning(f"Failed to answer callback: {e}")

# ─── ARTICLE MAP ────────────────────────────────────────────────
def load_article_map():
    try:
        with open("article_map.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        log.warning("article_map.json not found — run delivery.py first")
        return {}

# ─── DB ─────────────────────────────────────────────────────────
def save_feedback(article_id, vote, article_map):
    article = article_map.get(article_id, {})
    url = article.get("url", "")
    title = article.get("title", "")
    topic = article.get("topic", "")
    category = article.get("category", "")

    if not url:
        log.warning(f"No URL found for article_id {article_id} — skipping save")
        return

    try:
        conn = sqlite3.connect("pipeline.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO feedback (url, title, topic, category, vote, created_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
        """, (url, title, topic, category, vote))
        conn.commit()
        conn.close()
        emoji = "👍" if vote == "up" else "👎"
        log.info(f"Feedback saved: {emoji} — {title[:60]}")
    except Exception as e:
        log.error(f"DB error saving feedback: {e}")

# ─── MAIN LOOP ───────────────────────────────────────────────────
def main():
    log.info("Feedback listener started — waiting for button taps...")
    offset = None
    article_map = load_article_map()

    while True:
        try:
            updates = get_updates(offset)

            for update in updates.get("result", []):
                offset = update["update_id"] + 1

                cb = update.get("callback_query")
                if not cb:
                    continue

                data = cb.get("data", "")
                if not data.startswith("fb_"):
                    continue

                # Parse fb_{article_id}_{vote}
                parts = data.split("_")
                if len(parts) != 3:
                    continue

                _, article_id, vote = parts
                answer_callback(cb["id"])
                save_feedback(article_id, vote, article_map)

        except KeyboardInterrupt:
            log.info("Listener stopped.")
            break
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()