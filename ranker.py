import json
import os
import csv
import re
import pickle
from datetime import datetime, timezone
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import logging

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

# ─── HEURISTIC SCORING ──────────────────────────────────────────
HIGH_KEYWORDS = [
    "india", "modi", "delhi", "mumbai", "bangalore", "chennai",
    "pcos", "endometriosis", "women's health", "glp-1", "ozempic",
    "ai", "artificial intelligence", "openai", "gemini", "anthropic",
    "fashion week", "vogue", "cannes", "met gala",
    "breaking", "urgent", "exclusive", "crisis", "war", "election"
]

TRUSTED_SOURCES = [
    "reuters", "bbc", "the guardian", "associated press", "ap news",
    "the hindu", "ndtv", "healthline", "techcrunch", "the verge",
    "vogue", "allure", "pitchfork", "rolling stone"
]

SPAM_KEYWORDS = [
    "sponsored", "promoted", "advertisement", "buy now",
    "click here", "limited time offer", "market report", "cagr",
    "billion by 203", "million by 203"
]

def calculate_recency_score(published_at):
    """Newer articles score higher. Max 10 points."""
    try:
        if not published_at:
            return 0
        pub_time = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        hours_old = (now - pub_time).total_seconds() / 3600
        if hours_old <= 3:
            return 10
        elif hours_old <= 6:
            return 8
        elif hours_old <= 12:
            return 5
        elif hours_old <= 24:
            return 2
        else:
            return 0
    except Exception:
        return 0

def calculate_heuristic_score(article):
    """Calculate rule-based priority score."""
    score = 0
    title = article.get("title", "").lower()
    source = article.get("source", "").lower()
    description = article.get("description", "").lower()
    combined = title + " " + description

    # Keyword boost
    for keyword in HIGH_KEYWORDS:
        if keyword in combined:
            score += 8

    # Trusted source boost
    for trusted in TRUSTED_SOURCES:
        if trusted in source:
            score += 5
            break

    # Recency boost
    score += calculate_recency_score(article.get("published_at", ""))

    # Spam penalty
    for spam in SPAM_KEYWORDS:
        if spam in combined:
            score -= 15

    # Length penalty — very short titles are often clickbait
    if len(article.get("title", "")) < 20:
        score -= 5

    return score

# ─── LOAD + CLEAN LABELED DATA ──────────────────────────────────
def load_labeled_data(filepath="labeled_headlines.csv"):
    """Load and clean the labeled headlines CSV."""
    titles, labels = [], []
    label_map = {"high": 2, "medium": 1, "low": 0}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) < 2:
                    continue
                title = row[0].strip().strip('"').strip("'")
                priority = row[-1].strip().lower().strip('"').strip("'")

                # Clean up common formatting issues
                priority = re.sub(r'[^a-z]', '', priority)

                if priority in label_map and len(title) > 5:
                    titles.append(title)
                    labels.append(label_map[priority])

        log.info(f"Loaded {len(titles)} labeled headlines")
        return titles, labels

    except FileNotFoundError:
        log.error("labeled_headlines.csv not found")
        return [], []

# ─── TRAIN CLASSIFIER ───────────────────────────────────────────
def train_classifier(titles, labels):
    """Train TF-IDF + Logistic Regression classifier."""
    if len(titles) < 20:
        log.warning("Not enough labeled data to train classifier")
        return None, None

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words='english'
    )

    X = vectorizer.fit_transform(titles)

    # Train/test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    classifier.fit(X_train, y_train)

    # Evaluation metrics
    y_pred = classifier.predict(X_test)
    log.info("\n" + classification_report(y_test, y_pred,
             target_names=["Low", "Medium", "High"]))
    log.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Save model for reuse
    with open("classifier.pkl", "wb") as f:
        pickle.dump((vectorizer, classifier), f)
    log.info("Classifier saved to classifier.pkl")

    return vectorizer, classifier

# ─── RANK ARTICLES ──────────────────────────────────────────────
def rank_articles(articles, vectorizer, classifier):
    """Score each article combining ML + heuristics."""
    scored = []
    label_map = {0: "Low", 1: "Medium", 2: "High"}

    for article in articles:
        title = article.get("title", "")
        if not title:
            continue

        # ML score (0-2 mapped to 0-100)
        ml_score = 0
        if vectorizer and classifier:
            try:
                X = vectorizer.transform([title])
                proba = classifier.predict_proba(X)[0]
                ml_score = proba[2] * 60 + proba[1] * 30  # weighted
            except Exception as e:
                log.warning(f"ML scoring failed for '{title}': {e}")

        # Heuristic score
        heuristic_score = calculate_heuristic_score(article)

        # Combined score
        total_score = ml_score + heuristic_score

        article["ml_score"] = round(ml_score, 2)
        article["heuristic_score"] = round(heuristic_score, 2)
        article["total_score"] = round(total_score, 2)
        scored.append(article)

    # Sort by total score descending
    scored.sort(key=lambda x: x["total_score"], reverse=True)
    return scored

# ─── MAIN ────────────────────────────────────────────────────────
def main():
    start_time = datetime.now()
    log.info("Starting ranker pipeline...")

    # Load raw articles
    try:
        with open("raw_news.json", "r", encoding="utf-8") as f:
            articles = json.load(f)
        log.info(f"Loaded {len(articles)} articles from raw_news.json")
    except FileNotFoundError:
        log.error("raw_news.json not found. Run ingest.py first.")
        return

    # Train classifier
    titles, labels = load_labeled_data()
    vectorizer, classifier = train_classifier(titles, labels)

    # Rank articles
    ranked = rank_articles(articles, vectorizer, classifier)

    # Stats
    high_count = sum(1 for a in ranked if a["total_score"] >= 50)
    log.info(f"Articles scored: {len(ranked)}")
    log.info(f"High priority articles (score >= 50): {high_count}")
    log.info(f"Top article: '{ranked[0]['title']}' (score: {ranked[0]['total_score']})")

    # Save top 20
    top_articles = ranked[:20]
    with open("ranked_news.json", "w", encoding="utf-8") as f:
        json.dump(top_articles, f, indent=2, ensure_ascii=False)

    elapsed = (datetime.now() - start_time).total_seconds()
    log.info(f"Ranked {len(ranked)} articles in {elapsed:.2f} seconds")
    log.info(f"Top 15 saved to ranked_news.json")

    # Print top 5 for quick review
    print("\n── TOP 5 ARTICLES ──────────────────────────────")
    for i, a in enumerate(ranked[:5], 1):
        print(f"{i}. [{a['total_score']:.0f}] {a['title']}")
    print("────────────────────────────────────────────────\n")

if __name__ == "__main__":
    main()
