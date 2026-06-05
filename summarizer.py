import json
import os
import time
import logging
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

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

# ─── CONFIGURE GROQ ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

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
- Organize into these sections (only include sections that have relevant stories):
  🌍 World & Politics
  🤖 Tech & AI
  💄 Fashion & Beauty
  💪 Health & Wellness
  🇮🇳 India
  🎵 Music

Format each section like this:
## [emoji] [Section Name]
• [story summary]
• [story summary]
• [story summary]

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
        score = a.get("total_score", 0)
        url = a.get("url", "")

        article_list += f"""
Article {i} [ref:{i}]:
Title: {title}
Description: {description[:200] if description else 'N/A'}
Source: {source}
Topic: {topic}
Priority Score: {score}
---"""

    prompt = f"""Today is {today}.

Here are today's top ranked news articles. Generate Sidra's morning briefing.
When referencing an article in a bullet, end the bullet with [ref:N] where N is the article number.

{article_list}

Remember: Sharp, witty, personal. She wants signal not noise."""

    return prompt

# ─── STITCH URLS ────────────────────────────────────────────────
def stitch_urls(briefing, articles):
    for i, article in enumerate(articles, 1):
        url = article.get("url", "")
        if url:
            briefing = briefing.replace(f"[ref:{i}]", f"[Read more]({url})")
        else:
            briefing = briefing.replace(f"[ref:{i}]", "")
    return briefing

# ─── GENERATE BRIEFING ──────────────────────────────────────────
def generate_briefing(articles):
    if not articles:
        log.error("No articles to summarize")
        return None

    log.info(f"Sending {len(articles)} articles to Groq...")

    try:
        prompt = build_prompt(articles)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500,
        )

        briefing = response.choices[0].message.content
        briefing = stitch_urls(briefing, articles)
        log.info("Briefing generated successfully")
        return briefing

    except Exception as e:
        log.error(f"Groq API error: {e}")
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

    top_articles = articles[:15]
    log.info(f"Using top {len(top_articles)} articles for briefing")

    briefing = generate_briefing(top_articles)

    if not briefing:
        log.error("Failed to generate briefing")
        return

    today = datetime.now().strftime("%A, %B %d, %Y")
    full_briefing = f"""
       SIDRA'S MORNING BRIEFING
       {today}
-----------------------------


{briefing}

-----------------------------


─────────────────────────────────────
Generated at {datetime.now().strftime("%H:%M")} UAE time
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
