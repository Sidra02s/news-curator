import requests
import feedparser
import json
import os
import re
import sqlite3
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ─── YOUR API KEYS ───────────────────────────────────────────────
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY")

# ─── YOUR TOPICS (NewsAPI) ───────────────────────────────────────
NEWSAPI_TOPICS = [
    "fashion trends 2025",
    "skincare beauty makeup",
    "artificial intelligence technology",
    "PCOS endometriosis women health",
    "India politics Modi",
    "new music album release",
]

# ─── GUARDIAN SECTIONS ───────────────────────────────────────────
GUARDIAN_SECTIONS = [
    "fashion",
    "technology",
    "india",
    "music",
    "lifeandstyle",
    "world",
]

# ─── RSS FEEDS BY TOPIC ──────────────────────────────────────────
RSS_FEEDS = [
    # Fashion & Beauty
    ("https://www.vogue.com/feed/rss", "Fashion"),
    ("https://www.allure.com/feed/rss", "Beauty"),
    ("https://www.byrdie.com/feeds/all.rss", "Skincare"),

    # Women's Health
    ("https://www.healthline.com/rss/health-news", "Women's Health"),
    ("https://www.medicalnewstoday.com/rss/womens-health", "Women's Health"),

    # Indian Politics & News
    ("https://feeds.feedburner.com/ndtvnews-india-news", "India"),
    ("https://www.thehindu.com/news/national/feeder/default.rss", "India"),

    # Music
    ("https://pitchfork.com/rss/news/", "Music"),
    ("https://www.rollingstone.com/music/music-news/feed/", "Music"),

    # Tech & AI
    ("https://techcrunch.com/feed/", "Tech"),
    ("https://www.theverge.com/rss/index.xml", "Tech"),
    ("https://feeds.feedburner.com/AIWeekly", "AI"),
]

# ─── HTML CLEANER ────────────────────────────────────────────────
def strip_html(text):
    """Remove HTML tags and clean up whitespace from a string."""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─── FETCHERS ────────────────────────────────────────────────────

def fetch_newsapi(topic, api_key):
    if not api_key:
        print("  [NewsAPI] No API key found — skipping")
        return []
    url = "https://newsapi.org/v2/everything"
    yesterday = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    params = {
        "q": topic,
        "from": yesterday,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 10,
        "apiKey": api_key,
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        results = []
        for a in articles:
            if not a.get("title") or a.get("title") == "[Removed]":
                continue
            results.append({
                "title": strip_html(a.get("title", "").strip()),
                "description": strip_html(a.get("description", "")),
                "url": a.get("url", ""),
                "source": a.get("source", {}).get("name", "Unknown"),
                "published_at": a.get("publishedAt", ""),
                "topic": topic,
                "origin": "newsapi",
            })
        return results
    except Exception as e:
        print(f"  [NewsAPI] Error on '{topic}': {e}")
        return []


def fetch_guardian(section, api_key):
    if not api_key:
        print("  [Guardian] No API key found — skipping")
        return []
    url = "https://content.guardianapis.com/search"
    yesterday = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    params = {
        "section": section,
        "from-date": yesterday,
        "order-by": "newest",
        "page-size": 10,
        "api-key": api_key,
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        results_raw = response.json().get("response", {}).get("results", [])
        results = []
        for a in results_raw:
            results.append({
                "title": strip_html(a.get("webTitle", "").strip()),
                "description": "",
                "url": a.get("webUrl", ""),
                "source": "The Guardian",
                "published_at": a.get("webPublicationDate", ""),
                "topic": section,
                "origin": "guardian",
            })
        return results
    except Exception as e:
        print(f"  [Guardian] Error on section '{section}': {e}")
        return []


def fetch_rss(feed_url, topic_label):
    try:
        feed = feedparser.parse(feed_url)
        results = []
        for entry in feed.entries[:10]:
            title = strip_html(entry.get("title", "").strip())
            if not title:
                continue
            results.append({
                "title": title,
                "description": strip_html(entry.get("summary", "")),
                "url": entry.get("link", ""),
                "source": feed.feed.get("title", feed_url),
                "published_at": entry.get("published", ""),
                "topic": topic_label,
                "origin": "rss",
            })
        return results
    except Exception as e:
        print(f"  [RSS] Error on '{feed_url}': {e}")
        return []


def deduplicate(articles):
    try:
        from rapidfuzz import fuzz
        use_fuzzy = True
    except ImportError:
        print("  [Warning] rapidfuzz not installed, falling back to exact dedup.")
        use_fuzzy = False

    seen_titles = []
    unique = []
    duplicates_caught = 0

    for article in articles:
        title = article["title"].lower().strip()
        if not title:
            continue

        if not use_fuzzy:
            if title not in seen_titles:
                seen_titles.append(title)
                unique.append(article)
            continue

        is_duplicate = False
        for seen in seen_titles:
            score = fuzz.token_sort_ratio(title, seen)
            if score >= 85:
                is_duplicate = True
                duplicates_caught += 1
                break

        if not is_duplicate:
            seen_titles.append(title)
            unique.append(article)

    if use_fuzzy and duplicates_caught > 0:
        print(f"  [Fuzzy Dedup] Caught {duplicates_caught} near-duplicate articles")

    return unique


def save_articles_to_db(articles_list):
    connection = sqlite3.connect("pipeline.db")
    cursor = connection.cursor()
    saved_count = 0

    for article in articles_list:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO articles (title, url, source, published_at, priority)
                VALUES (?, ?, ?, ?, ?)
            """, (
                article.get('title'),
                article.get('url'),
                article.get('source'),
                article.get('published_at'),
                'Low'
            ))
            if cursor.rowcount > 0:
                saved_count += 1
        except Exception as e:
            print(f"Error inserting article: {e}")
            continue

    connection.commit()
    connection.close()
    print(f"Saved {saved_count} new articles to database")


def main():
    all_articles = []

    print("\n[1/3] Fetching from NewsAPI...")
    if not NEWS_API_KEY:
        print("  WARNING: NEWS_API_KEY not found in .env — skipping NewsAPI")
    for topic in NEWSAPI_TOPICS:
        articles = fetch_newsapi(topic, NEWS_API_KEY)
        print(f"  '{topic}': {len(articles)} articles")
        all_articles.extend(articles)

    print("\n[2/3] Fetching from The Guardian...")
    if not GUARDIAN_API_KEY:
        print("  WARNING: GUARDIAN_API_KEY not found in .env — skipping Guardian")
    for section in GUARDIAN_SECTIONS:
        articles = fetch_guardian(section, GUARDIAN_API_KEY)
        print(f"  Section '{section}': {len(articles)} articles")
        all_articles.extend(articles)

    print("\n[3/3] Fetching from RSS feeds...")
    for feed_url, topic_label in RSS_FEEDS:
        articles = fetch_rss(feed_url, topic_label)
        print(f"  '{topic_label}' ({feed_url.split('/')[2]}): {len(articles)} articles")
        all_articles.extend(articles)

    unique_articles = deduplicate(all_articles)
    print(f"\nTotal after deduplication: {len(unique_articles)} articles")

    with open("raw_news.json", "w", encoding="utf-8") as f:
        json.dump(unique_articles, f, indent=2, ensure_ascii=False)
    print("Saved to raw_news.json")

    save_articles_to_db(unique_articles)

    print(f"\nBreakdown by origin:")
    for origin in ["newsapi", "guardian", "rss"]:
        count = sum(1 for a in unique_articles if a["origin"] == origin)
        print(f"  {origin}: {count} articles")


if __name__ == "__main__":
    main()