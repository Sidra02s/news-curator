"""
feedback_to_csv.py — Reads thumbs up/down votes from pipeline.db
and merges them into labeled_headlines.csv for retraining.

Thumbs up (vote='up')  → High
Thumbs down (vote='dn') → Skip

Run before train_models.py to include user feedback in retraining.
"""

import sqlite3
import csv
import os

LABELED_CSV = "labeled_headlines.csv"
DB_PATH = "pipeline.db"


def load_existing_titles(filepath):
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


def main():
    print("Loading existing labeled titles...")
    existing = load_existing_titles(LABELED_CSV)
    print(f"  {len(existing)} titles already in CSV")

    print("Reading feedback votes from database...")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT title, vote FROM feedback WHERE title IS NOT NULL")
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"DB error: {e}")
        return

    if not rows:
        print("No feedback votes found in database. Nothing to merge.")
        return

    new_rows = []
    skipped = 0

    for title, vote in rows:
        title = title.strip()
        if not title or title.lower() in existing:
            skipped += 1
            continue

        if vote == "up":
            label = "Keep"
        elif vote == "dn":
            label = "Skip"
        else:
            skipped += 1
            continue

        new_rows.append((title, label))
        existing.add(title.lower())

    if not new_rows:
        print(f"No new feedback to add ({skipped} already existed or invalid).")
        return

    file_exists = os.path.exists(LABELED_CSV)
    with open(LABELED_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["title", "priority"])
        for title, label in new_rows:
            writer.writerow([title, label])

    print(f"Added {len(new_rows)} feedback votes to {LABELED_CSV}")
    print(f"Skipped {skipped} (duplicates or invalid)")
    print(f"Keep={sum(1 for _,l in new_rows if l=='Keep')}, Skip={sum(1 for _,l in new_rows if l=='Skip')}")
    print("Ready to retrain — run train_models.py")


if __name__ == "__main__":
    main()