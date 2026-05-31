import requests
import feedparser
import json
import os
import sqlite3
from datetime import datetime, timedelta

# ─── YOUR API KEYS ───────────────────────────────────────────────
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY")

# ─── YOUR TOPICS (NewsAPI) ───────────────────────────────────────
# These are searched across 150,000+ sources
NEWSAPI_TOPICS = [
    "fashion trends 2025",
    "skincare beauty makeup",
    "artificial intelligence technology",
    "PCOS endometriosis women health",
    "India politics Modi",
    "new music album release",
]

# ─── GUARDIAN SECTIONS ───────────────────────────────────────────
# Guardian has clean section-based filtering — much more reliable
# Full list: https://open-platform.theguardian.com/explore/
GUARDIAN_SECTIONS = [
    "fashion",          # Fashion & Beauty
    "technology",       # Tech & AI
    "india",            # Indian Politics
    "music",            # Music
    "lifeandstyle",     # Covers wellness, health, beauty
    "world",            # World news
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


# ─── FETCHERS ────────────────────────────────────────────────────

def fetch_newsapi(topic, api_key):
    """Fetch articles from NewsAPI for a given topic."""
    url = "https://newsapi.org/v2/everything"
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
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
                "title": a.get("title", "").strip(),
                "description": a.get("description", ""),
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
    """Fetch articles from The Guardian API for a given section."""
    url = "https://content.guardianapis.com/search"
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
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
                "title": a.get("webTitle", "").strip(),
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
    """Fetch articles from an RSS feed."""
    try:
        feed = feedparser.parse(feed_url)
        results = []
        for entry in feed.entries[:10]:
            title = entry.get("title", "").strip()
            if not title:
                continue
            results.append({
                "title": title,
                "description": entry.get("summary", ""),
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
    """Remove duplicate articles by normalizing titles."""
    seen = set()
    unique = []
    for article in articles:
        key = article["title"].lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(article)
    return unique


# ─── DATABASE STORAGE INTEGRATION ────────────────────────────────

def save_articles_to_db(articles_list):
    """Saves incoming pipeline metrics and headlines to SQLite database archive."""
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
            print(f"Error inserting article into SQLite execution context: {e}")
            continue

    connection.commit()
    connection.close()
    print(f"💾 Successfully saved {saved_count} brand new unique articles to SQLite database!")


# ─── MAIN EXECUTION PIPELINE ─────────────────────────────────────

def main():
    all_articles = []

    # 1. NewsAPI
    print("\n[1/3] Fetching from NewsAPI...")
    for topic in NEWSAPI_TOPICS:
        articles = fetch_newsapi(topic, NEWS_API_KEY)
        print(f"  '{topic}': {len(articles)} articles")
        all_articles.extend(articles)

    # 2. The Guardian
    print("\n[2/3] Fetching from The Guardian...")
    for section in GUARDIAN_SECTIONS:
        articles = fetch_guardian(section, GUARDIAN_API_KEY)
        print(f"  Section '{section}': {len(articles)} articles")
        all_articles.extend(articles)

    # 3. RSS Feeds
    print("\n[3/3] Fetching from RSS feeds...")
    for feed_url, topic_label in RSS_FEEDS:
        articles = fetch_rss(feed_url, topic_label)
        print(f"  '{topic_label}' ({feed_url.split('/')[2]}): {len(articles)} articles")
        all_articles.extend(articles)

    # Deduplicate
    unique_articles = deduplicate(all_articles)
    print(f"\n✓ Total after deduplication: {len(unique_articles)} articles")

    # Save to flat file legacy backup layer
    with open("raw_news.json", "w", encoding="utf-8") as f:
        json.dump(unique_articles, f, indent=2, ensure_ascii=False)
    print("✓ Saved to raw_news.json legacy log file")

    # Save directly to SQL Database Relational Storage Engine
    save_articles_to_db(unique_articles)

    print(f"\nBreakdown by origin:")
    for origin in ["newsapi", "guardian", "rss"]:
        count = sum(1 for a in unique_articles if a["origin"] == origin)
        print(f"  {origin}: {count} articles")


if __name__ == "__main__":
    main()