"""
autolabel.py — Expands labeled_headlines.csv using Groq to auto-label
articles from raw_news.json and pipeline.db.

Binary classification: High or Skip only.
Run once to bulk-generate labels, then retrain the classifier.
Usage: python autolabel.py
"""

import os
import csv
import json
import sqlite3
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

LABEL_PROMPT = """You are a news priority classifier for a personal morning briefing system.

The reader cares about these topics:
- India politics, social issues, government decisions
- AI, tech, startups, product launches
- Fashion, beauty, skincare, style trends
- Women's health: PCOS, endometriosis, hormones, GLP-1/Ozempic, mental health
- World politics, geopolitics, wars, elections
- Music: new releases, artist news, concerts

Classify each headline as either Keep or Skip using these rules:

KEEP: Breaking news, major policy decisions, significant scientific findings,
      important product launches, major political events, health breakthroughs,
      anything directly relevant to the reader's core interests listed above.
      Interesting trends, industry news, moderate celebrity news also qualify.

SKIP: Market reports, clickbait, listicles, sponsored content, routine
      announcements, topics completely outside the reader's interests,
      obscure artist news, CAGR reports, press releases.

You will receive a list of headlines numbered like:
1. Headline text
2. Headline text

Respond ONLY with a JSON array in this exact format, nothing else:
[{"id": 1, "label": "Keep"}, {"id": 2, "label": "Skip"}, ...]

Every id must appear exactly once. Labels must be exactly: Keep or Skip."""


def load_existing_titles(filepath="labeled_headlines.csv"):
    existing = set()
    try:
        with open(filepath, "r", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row:
                    existing.add(row[0].strip().lower())
    except FileNotFoundError:
        pass
    return existing


def collect_unlabeled_articles(existing_titles):
    candidates = []

    try:
        with open("raw_news.json", "r", encoding="utf-8") as f:
            articles = json.load(f)
        for a in articles:
            title = a.get("title", "").strip()
            if title and title.lower() not in existing_titles and len(title) > 15:
                candidates.append(title)
    except FileNotFoundError:
        print("raw_news.json not found, skipping")

    try:
        conn = sqlite3.connect("pipeline.db")
        cursor = conn.cursor()
        cursor.execute("SELECT title FROM articles WHERE title IS NOT NULL")
        rows = cursor.fetchall()
        conn.close()
        for (title,) in rows:
            title = title.strip()
            if title and title.lower() not in existing_titles and len(title) > 15:
                if title not in candidates:
                    candidates.append(title)
    except Exception as e:
        print(f"DB read error: {e}")

    seen = set()
    unique = []
    for t in candidates:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t)

    return unique


def label_batch(titles_batch):
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles_batch))

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": LABEL_PROMPT},
                    {"role": "user", "content": f"Headlines to classify:\n{numbered}"}
                ],
                temperature=0.1,
                max_tokens=2000,
            )

            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            results = json.loads(text)

            labels = {}
            for item in results:
                idx = item["id"] - 1
                label = item["label"].strip().capitalize()
                if label in ["Keep", "Skip"] and 0 <= idx < len(titles_batch):
                    labels[idx] = label

            return labels

        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(5)

    return {}


def append_to_csv(new_rows, filepath="labeled_headlines.csv"):
    file_exists = os.path.exists(filepath)
    with open(filepath, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["title", "priority"])
        for title, label in new_rows:
            writer.writerow([title, label])


def main():
    print("Loading existing labeled titles...")
    existing = load_existing_titles()
    print(f"  {len(existing)} titles already labeled")

    print("Collecting unlabeled articles...")
    candidates = collect_unlabeled_articles(existing)
    print(f"  {len(candidates)} new articles to label")

    if not candidates:
        print("Nothing new to label. Run ingest.py first to get fresh articles.")
        return

    batch_size = 50
    total_labeled = 0
    new_rows = []

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]
        print(f"  Labeling batch {i//batch_size + 1}/{(len(candidates)-1)//batch_size + 1} ({len(batch)} titles)...")

        labels = label_batch(batch)

        for idx, label in labels.items():
            new_rows.append((batch[idx], label))
            total_labeled += 1

        time.sleep(2)

    append_to_csv(new_rows)

    from collections import Counter
    dist = Counter(l for _, l in new_rows)
    print(f"\nDone. Added {total_labeled} new labeled articles.")
    print(f"Distribution of new labels: {dict(dist)}")
    print(f"Total in CSV now: {len(existing) + total_labeled}")
    print("\nNext step: run train_models.py to retrain the classifier.")


if __name__ == "__main__":
    main()