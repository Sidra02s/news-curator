import os
import re
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
from telegram import Bot
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

# ─── SEND MESSAGE ───────────────────────────────────────────────
async def send_briefing(text):
    """Send briefing to Telegram in chunks if too long."""
    text = clean_markdown(text)
    bot = Bot(token=TELEGRAM_TOKEN)

    max_length = 4000
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]

    try:
        for i, chunk in enumerate(chunks):
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=chunk,
                parse_mode=None
            )
            if len(chunks) > 1:
                log.info(f"Sent chunk {i+1} of {len(chunks)}")

        log.info("Briefing delivered to Telegram successfully")

    except TelegramError as e:
        log.error(f"Telegram error: {e}")
        raise

# ─── MAIN ────────────────────────────────────────────────────────
def main():
    log.info("Starting delivery...")

    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_TOKEN not found in .env")
        return
    if not TELEGRAM_CHAT_ID:
        log.error("TELEGRAM_CHAT_ID not found in .env")
        return

    try:
        with open("briefing.txt", "r", encoding="utf-8") as f:
            briefing = f.read()
        log.info(f"Loaded briefing ({len(briefing)} characters)")
    except FileNotFoundError:
        log.error("briefing.txt not found. Run summarizer.py first.")
        return

    asyncio.run(send_briefing(briefing))
    log.info("Delivery complete")

if __name__ == "__main__":
    main()