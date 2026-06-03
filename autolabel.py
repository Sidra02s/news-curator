"""
autolabel.py — Expands labeled_headlines.csv using Gemini to auto-label
articles from raw_news.json and pipeline.db.

Run once to bulk-generate labels, then retrain the classifier.
Usage: python autolabel.py
"""

import os
import csv
import json
import sqlite3
import time
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

LABEL_PROMPT = """You are a news priority classifier for a personal morning briefing system.

The reader cares about these topics:
- India politics, social issues, government decisions
- AI, tech, startups, product launches
- Fashion, beauty, skincare, style trends
- Women's health: PCOS, endometriosis, hormones, GLP-1/Ozempic, mental health
- World politics, geopolitics, wars, elections
- Music: new releases, artist news, concerts

Classify each headline as High, Medium, or Low priority using these rules:

HIGH: Breaking news, major policy decisions, significant scientific findings, 
      important product launches, major political events, health breakthroughs,
      anything directly relevant to the reader's core interests listed above.

MEDIUM: Interesting but not urgent. Industry trends, opinion pieces, moderate 
        celebrity news, regional news, follow-up stories.

LOW: Market reports, clickbait, listicles, obscure artist news, 
     sponsored content, routine announcements, album reviews of unknown artists,
     topics completely outside the reader's interests.

You will receive a list of headlines numbered like:
1. Headline text
2. Headline text

Respond ONLY with a JSON array in this exact format, nothing else, no markdown:
[{"id": 1, "label": "High"}, {"id": 2, "label": "Low"}, ...]

Every id must appear exactly once. Labels must be exactly: High, Medium, or Low."""


def load_existing_titles(filepath="labeled_headlines.csv"):
    existing = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
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


def parse_labels(text, batch_size):
    """Robustly parse Gemini's JSON response, handling common malformations."""
    # Strip markdown fences
    text = re.sub(r'```json|```', '', text).strip()

    # Try direct parse first
    try:
        results = json.loads(text)
        labels = {}
        for item in results:
            idx = item["id"] - 1
            label = item["label"].strip().capitalize()
            if label in ["High", "Medium", "Low"] and 0 <= idx < batch_size:
                labels[idx] = label
        return labels
    except Exception:
        pass

    # Fallback: extract individual {"id": X, "label": "Y"} pairs with regex
    labels = {}
    pattern = r'"id"\s*:\s*(\d+)[^}]*"label"\s*:\s*"([^"]+)"'
    matches = re.findall(pattern, text)
    for id_str, label in matches:
        idx = int(id_str) - 1
        label = label.strip().capitalize()
        if label in ["High", "Medium", "Low"] and 0 <= idx < batch_size:
            labels[idx] = label

    return labels


def label_batch(titles_batch):
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles_batch))

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{LABEL_PROMPT}\n\nHeadlines to classify:\n{numbered}",
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=2000,
                )
            )

            text = response.text.strip()
            labels = parse_labels(text, len(titles_batch))

            if len(labels) == 0:
                raise Exception("No valid labels parsed from response")

            return labels

        except Exception as e:
            wait = 15 * (attempt + 1)  # 15s, 30s, 45s, 60s, 75s
            print(f"  Attempt {attempt+1} failed: {e}")
            if attempt < 4:
                print(f"  Waiting {wait}s before retry...")
                time.sleep(wait)

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

    batch_size = 20
    total_labeled = 0
    new_rows = []

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(candidates)-1)//batch_size + 1
        print(f"  Labeling batch {batch_num}/{total_batches} ({len(batch)} titles)...")

        labels = label_batch(batch)

        if labels:
            for idx, label in labels.items():
                new_rows.append((batch[idx], label))
                total_labeled += 1
            print(f"    Got {len(labels)} labels")
        else:
            print(f"    Batch {batch_num} failed completely, skipping")

        # Rate limit pause between batches
        time.sleep(3)

    append_to_csv(new_rows)

    from collections import Counter
    dist = Counter(l for _, l in new_rows)
    print(f"\nDone. Added {total_labeled} new labeled articles.")
    print(f"Distribution of new labels: {dict(dist)}")
    print(f"Total in CSV now: {len(existing) + total_labeled}")
    print("\nNext step: run train_models.py to retrain on expanded dataset.")


if __name__ == "__main__":
    main()