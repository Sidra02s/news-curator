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
- You MUST include at least 4 different sections every time
- Organize into these sections:
  🌍 World & Politics
  🤖 Tech & AI
  💄 Fashion & Beauty
  💪 Health & Wellness
  🇮🇳 India
  🎵 Music

Format each section like this:
*[emoji] [Section Name]*
• [story summary]
• [story summary]
• [story summary]

End with a one-line "Today's Takeaway" that captures the most important thing happening in the world right now."""

# ─── BUILD PROMPT ───────────────────────────────────────────────
def build_prompt(articles):
    """Format top articles into a prompt."""
    today = datetime.now().strftime("%A, %B %d, %Y")
    article_list = ""

    for i, a in enumerate(articles, 1):
        title = a.get("title", "")
        description = a.get("description", "") or ""
        source = a.get("source", "Unknown")
        topic = a.get("topic", "")
        score = a.get("total_score", 0)

        article_list += f"""
Article {i}:
Title: {title}
Description: {description[:200] if description else 'N/A'}
Source: {source}
Topic: {topic}
Priority Score: {score}
---"""

    prompt = f"""Today is {today}.

Here are today's top ranked news articles. Generate Sidra's morning briefing:

{article_list}

Remember: Sharp, witty, personal. She wants signal not noise."""

    return prompt

# ─── GENERATE BRIEFING ──────────────────────────────────────────
def generate_briefing(articles):
    """Send articles to Gemini and get briefing back with retry logic."""
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
                    max_output_tokens=1500,
                    )
            )

            briefing = response.text
            log.info("Briefing generated successfully")
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

    # Load ranked articles
    try:
        with open("ranked_news.json", "r", encoding="utf-8") as f:
            articles = json.load(f)
        log.info(f"Loaded {len(articles)} ranked articles")
    except FileNotFoundError:
        log.error("ranked_news.json not found. Run ranker.py first.")
        return

    # Take top 20
    top_articles = articles[:20]
    log.info(f"Using top {len(top_articles)} articles for briefing")

    # Generate briefing
    briefing = generate_briefing(top_articles)

    if not briefing:
        log.error("Failed to generate briefing")
        return

    # ─── TIMEZONE-AWARE HEADER GENERATION ────────────────────────────
    # Set the timezone explicitly to UAE (Asia/Dubai) for GitHub Actions
    uae_tz = ZoneInfo("Asia/Dubai")
    now_uae = datetime.now(uae_tz)

    today = now_uae.strftime("%A, %B %d, %Y")
    generated_time = now_uae.strftime("%H:%M")

    full_briefing = f"""🗞 *SIDRA'S MORNING BRIEFING*
📅 {today}
─────────────────────────────

{briefing}

─────────────────────────────
⏰ Generated at {generated_time} UAE time
"""
    # ──────────────────────────────────────────────────────────────────

    # Save to file
    with open("briefing.txt", "w", encoding="utf-8") as f:
        f.write(full_briefing)

    elapsed = (datetime.now() - start_time).total_seconds()
    log.info(f"Briefing saved to briefing.txt in {elapsed:.2f} seconds")

    # Print preview
    print("\n" + "="*50)
    print(full_briefing)
    print("="*50 + "\n")

if __name__ == "__main__":
    main()