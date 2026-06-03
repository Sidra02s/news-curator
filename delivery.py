import os
import re
import json
import asyncio
import hashlib
import logging
from datetime import datetime
from dotenv import load_dotenv
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import TelegramError

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

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ─── CLEAN MARKDOWN ─────────────────────────────────────────────
def clean_markdown(text):
    """Strip all markdown symbols from text."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    text = re.sub(r'#{1,6}\s', '', text)
    return text

# ─── ARTICLE MAP ────────────────────────────────────────────────
def get_article_id(url):
    """Short MD5 hash to identify an article within Telegram's 64-byte callback_data limit."""
    return hashlib.md5(url.encode()).hexdigest()[:8]

def save_article_map(articles):
    """Save article_id → metadata mapping so feedback_listener can look up votes."""
    article_map = {}
    for a in articles:
        article_id = get_article_id(a.get("url", ""))
        article_map[article_id] = {
            "url": a.get("url", ""),
            "title": a.get("title", ""),
            "topic": a.get("topic", ""),
            "category": a.get("category", "")
        }
    with open("article_map.json", "w", encoding="utf-8") as f:
        json.dump(article_map, f, indent=2)
    log.info(f"Saved article map with {len(article_map)} entries")

# ─── SEND BRIEFING TEXT ─────────────────────────────────────────
async def send_briefing(text, bot):
    """Send the main briefing text to Telegram in chunks if too long."""
    text = clean_markdown(text)

    max_length = 4000
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]

    for i, chunk in enumerate(chunks):
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=chunk,
            parse_mode=None,
            connect_timeout=30,
            read_timeout=30,
            write_timeout=30
        )
        if len(chunks) > 1:
            log.info(f"Sent chunk {i+1} of {len(chunks)}")

    log.info("Briefing text delivered successfully")

# ─── SEND ARTICLE BUTTONS ───────────────────────────────────────
async def send_article_buttons(articles, bot):
    """Send each article as a separate message with 👍👎 inline buttons."""
    for a in articles:
        url = a.get("url", "")
        title = a.get("title", "No title")
        description = (a.get("description", "") or "")[:120]
        article_id = get_article_id(url)

        text = f"{title}"
        if description:
            text += f"\n{description}"
        if url:
            text += f"\n{url}"

        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("👍 Relevant", callback_data=f"fb_{article_id}_up"),
            InlineKeyboardButton("👎 Skip", callback_data=f"fb_{article_id}_dn")
        ]])

        try:
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=text,
                reply_markup=keyboard,
                parse_mode=None,
                connect_timeout=30,
                read_timeout=30,
                write_timeout=30
            )
        except TelegramError as e:
            log.warning(f"Failed to send buttons for '{title}': {e}")

    log.info(f"Sent {len(articles)} articles with feedback buttons")

# ─── MAIN ────────────────────────────────────────────────────────
def main():
    log.info("Starting delivery...")

    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_TOKEN not found in .env")
        return
    if not TELEGRAM_CHAT_ID:
        log.error("TELEGRAM_CHAT_ID not found in .env")
        return

    # Load briefing text
    try:
        with open("briefing.txt", "r", encoding="utf-8") as f:
            briefing = f.read()
        log.info(f"Loaded briefing ({len(briefing)} characters)")
    except FileNotFoundError:
        log.error("briefing.txt not found. Run summarizer.py first.")
        return

    # Load ranked articles for buttons
    try:
        with open("ranked_news.json", "r", encoding="utf-8") as f:
            articles = json.load(f)
        top_articles = articles[:15]
        log.info(f"Loaded {len(top_articles)} articles for feedback buttons")
    except FileNotFoundError:
        log.error("ranked_news.json not found. Run ranker.py first.")
        return

    # Save article map for feedback_listener
    save_article_map(top_articles)

    async def run():
        bot = Bot(token=TELEGRAM_TOKEN)
        await send_briefing(briefing, bot)
        await send_article_buttons(top_articles, bot)

    asyncio.run(run())
    log.info("Delivery complete")

if __name__ == "__main__":
    main()