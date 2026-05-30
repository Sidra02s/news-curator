import json
import os
import time
import logging
import sqlite3  # Added database driver
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
# Pull API Key safely out of environment state context
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
- Do NOT use any markdown symbols like **, *, #, or __
- Plain text only — no bold, no italic, no headers with symbols
- After each bullet point include the source URL in brackets like this:[Read more](url)
- You MUST include ALL of these sections every time:
  🌍 World & Politics
  🤖 Tech & AI
  💄 Fashion & Beauty
  💪 Health & Wellness
  🇮🇳 India
  🎵 Music

Format each section exactly like this:
🌍 World & Politics
- [story summary]
- [story summary]
- [story summary]

End with:
TODAY'S TAKEAWAY: [one line capturing the most important thing happening right now]"""

# ─── DATABASE FETCH FUNCTION ────────────────────────────────────
def fetch_high_priority_articles_from_db():
    """Queries the SQLite database relational layer for the top freshest High priority rows."""
    try:
        connection = sqlite3.connect("pipeline.db")
        cursor = connection.cursor()
        
        # Pulling the top 20 high signal headlines
        cursor.execute("""
            SELECT title, source, url 
            FROM articles 
            WHERE priority = 'High' 
            ORDER BY id DESC 
            LIMIT 20
        """)
        rows = cursor.fetchall()
        connection.close()
        
        articles = []
        for title, source, url in rows:
            articles.append({
                "title": title,
                "description": "", # Default blank since database tracks headlines
                "source": source,
                "url": url,
                "category": "High Signal News Archive"
            })
        return articles
    except Exception as e:
        log.error(f"Database collection failed: {e}")
        return []

# ─── BUILD PROMPT ───────────────────────────────────────────────
def build_prompt(articles):
    today = datetime.now().strftime("%A, %B %d, %Y")
    article_list = ""

    for i, a in enumerate(articles, 1):
        title = a.get("title", "")
        description = a.get("description", "") or ""
        source = a.get("source", "Unknown")
        category = a.get("category", "")
        url = a.get("url", "")
        
        article_list += f"""
Article {i} [Category: {category}]:
Title: {title}
Description: {description[:200] if description else 'N/A'}
Source: {source}
URL: {url}
---"""
    prompt = f"""Today is {today}.

Here are today's top ranked news articles organized by category.
Generate Sidra's complete morning briefing covering ALL 6 sections.
No markdown symbols. Plain text only.

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

            if len(briefing) < 1000:
               log.warning(f"Briefing too short ({len(briefing)} chars) — likely truncated. Retrying...")
               raise Exception("Briefing output too short")

            log.info(f"Briefing generated ({len(briefing)} characters)")
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
    log.info("Starting database-integrated summarizer...")

    # Swapped from opening ranked_news.json to fetching straight from SQLite columns
    top_articles = fetch_high_priority_articles_from_db()
    log.info(f"Loaded {len(top_articles)} high-priority articles from SQL engine")

    if not top_articles:
        log.error("No high priority data rows found in database context. Make sure to execute ranker.py first.")
        return

    briefing = generate_briefing(top_articles)

    if not briefing:
        log.error("Failed to generate briefing string output matching credentials.")
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