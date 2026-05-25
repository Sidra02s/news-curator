import json
import os
import time
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from google import genai
from google.genai import types

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

# ─── CONFIGURE GEMINI ───────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# ─── SYSTEM PROMPT ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are Sidra's personal news editor — sharp, direct, and witty.
Your job is to turn today's top news into a punchy morning briefing she'll actually want to read.

Rules:
- Be concise. No fluff. No filler.
- Use bullet points — max 3 per section
- Each bullet is 1-2 sentences max
- Be opinionated where relevant — this is a personal briefing, not a newspaper
- If a story is genuinely important, say why
- You MUST include ALL of these sections every time, even if you have to pull from lower ranked articles:
  🌍 World & Politics
  🤖 Tech & AI
  💄 Fashion & Beauty
  💪 Health & Wellness
  🇮🇳 India
  🎵 Music
Format each section exactly like this:
[emoji + SECTION NAME]:
- [story summary]
- [story summary]
- [story summary]

End with:
TODAY'S TAKEAWAY: [one line capturing the most important thing happening right now]"""

# ─── BUILD PROMPT ───────────────────────────────────────────────
def build_prompt(articles):
    today = datetime.now().strftime("%A, %B %d, %Y")
    article_list = ""

    for i, a in enumerate(articles, 1):
        title = a.get("title", "")
        description = a.get("description", "") or ""
        source = a.get("source", "Unknown")
        topic = a.get("topic", "")
        category = a.get("category", "")
        score = a.get("total_score", 0)

        article_list += f"""
Article {i} [Category: {category}]:
Title: {title}
Description: {description[:200] if description else 'N/A'}
Source: {source}
---"""

    prompt = f"""Today is {today}.

Here are today's top ranked news articles organized by category.
Generate Sidra's complete morning briefing covering ALL 6 sections.
If a section has no articles from the top ranked list, still include it with 1-2 relevant stories from the list.

{article_list}

IMPORTANT: Write the COMPLETE briefing with ALL 6 sections. Do not stop early."""

    return prompt

# ─── GENERATE BRIEFING ──────────────────────────────────────────
def generate_briefing(articles):
    if not articles:
        log.error("No articles to summarize")
        return None

    log.info(f"Sending {len(articles)} articles to Gemini...")

    for attempt in range(3):
        try:
            prompt = build_prompt(articles)

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=3000,
                )
            )

            briefing = response.text
            log.info(f"Briefing generated successfully ({len(briefing)} characters)")
            return briefing

        except Exception as e:
            log.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                log.info("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                log.error("All 3 attempts failed")
                return None

# ─── MAIN ────────────────────────────────────────────────────────
def main():
    start_time = datetime.now()
    log.info("Starting summarizer...")

    try:
        with open("ranked_news.json", "r", encoding="utf-8") as f:
            articles = json.load(f)
        log.info(f"Loaded {len(articles)} ranked articles")
    except FileNotFoundError:
        log.error("ranked_news.json not found. Run ranker.py first.")
        return

    top_articles = articles[:20]
    log.info(f"Using top {len(top_articles)} articles for briefing")

    briefing = generate_briefing(top_articles)

    if not briefing:
        log.error("Failed to generate briefing")
        return

    uae_tz = ZoneInfo("Asia/Dubai")
    now_uae = datetime.now(uae_tz)
    today = now_uae.strftime("%A, %B %d, %Y")
    generated_time = now_uae.strftime("%H:%M")

    full_briefing = f"""SIDRA'S MORNING BRIEFING
{today}
-----------------------------

{briefing}

-----------------------------
Generated at {generated_time} UAE time
"""

    with open("briefing.txt", "w", encoding="utf-8") as f:
        f.write(full_briefing)

    elapsed = (datetime.now() - start_time).total_seconds()
    log.info(f"Briefing saved to briefing.txt in {elapsed:.2f} seconds")

    print("\n" + "="*50)
    print(full_briefing)
    print("="*50 + "\n")

if __name__ == "__main__":
    main()