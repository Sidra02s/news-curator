import json
import os
import csv
import re
import pickle
import sqlite3
from datetime import datetime, timezone
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

# ─── TOPIC MAPPING ──────────────────────────────────────────────
TOPIC_CATEGORIES = {
    "fashion": ["fashion trends 2025", "fashion", "vogue", "allure", "skincare"],
    "health": ["PCOS endometriosis women health", "Women's Health"],
    "tech": ["artificial intelligence technology", "Tech", "AI"],
    "india": ["India politics Modi", "India"],
    "music": ["new music album release", "Music"],
    "world": ["world", "lifeandstyle"],
    "beauty": ["skincare beauty makeup", "Skincare", "Beauty"],
}

def get_topic_category(article):
    topic = article.get("topic", "").lower()
    source = article.get("source", "").lower()
    for category, keywords in TOPIC_CATEGORIES.items():
        for keyword in keywords:
            if keyword.lower() in topic or keyword.lower() in source:
                return category
    return "other"

def calculate_recency_score(published_at):
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
    score = 0
    title = article.get("title", "").lower()
    source = article.get("source", "").lower()
    description = article.get("description", "").lower()
    combined = title + " " + description

    for keyword in HIGH_KEYWORDS:
        if keyword in combined:
            score += 8

    for trusted in TRUSTED_SOURCES:
        if trusted in source:
            score += 5
            break

    score += calculate_recency_score(article.get("published_at", ""))

    for spam in SPAM_KEYWORDS:
        if spam in combined:
            score -= 15

    if len(article.get("title", "")) < 20:
        score -= 5

    return score

def load_labeled_data(filepath="labeled_headlines.csv"):
    titles, labels = [], []
    label_map = {"high": 2, "medium": 1, "low": 0}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 2:
                    continue
                title = row[0].strip().strip('"').strip("'")
                priority = row[-1].strip().lower().strip('"').strip("'")
                priority = re.sub(r'[^a-z]', '', priority)

                if priority in label_map and len(title) > 5:
                    titles.append(title)
                    labels.append(label_map[priority])

        log.info(f"Loaded {len(titles)} labeled headlines")
        return titles, labels

    except FileNotFoundError:
        log.error("labeled_headlines.csv not found")
        return [], []

def load_classifier():
    """Load classifier.pkl — supports logistic regression, naive bayes, and sentence transformer."""
    try:
        with open("classifier.pkl", "rb") as f:
            data = pickle.load(f)
        
        # New format: tuple with model type tag
        if isinstance(data, tuple) and isinstance(data[0], str):
            model_type = data[0]
            log.info(f"Loaded classifier type: {model_type}")
            return data
        
        # Legacy format: (vectorizer, classifier) tuple
        log.info("Loaded legacy classifier format")
        return ("logistic_regression_legacy", data[0], data[1])
    
    except FileNotFoundError:
        log.warning("classifier.pkl not found — will train from scratch")
        return None

def train_classifier(titles, labels):
    """Fallback: train logistic regression if no classifier.pkl exists."""
    if len(titles) < 20:
        log.warning("Not enough labeled data to train classifier")
        return None

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words='english'
    )

    from sklearn.pipeline import Pipeline
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    pipeline = Pipeline([('tfidf', vectorizer), ('clf', classifier)])

    X_train, X_test, y_train, y_test = train_test_split(
        titles, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    log.info("\n" + classification_report(y_test, y_pred,
             target_names=["Low", "Medium", "High"]))

    import pickle
    with open("classifier.pkl", "wb") as f:
        pickle.dump(("logistic_regression", pipeline), f)
    log.info("Classifier saved to classifier.pkl")

    return ("logistic_regression", pipeline)

def get_ml_score(title, classifier_data):
    """Get ML score for a title regardless of model type."""
    if classifier_data is None:
        return 0

    model_type = classifier_data[0]

    try:
        if model_type in ("logistic_regression", "naive_bayes"):
            pipeline = classifier_data[1]
            proba = pipeline.predict_proba([title])[0]
            return proba[2] * 60 + proba[1] * 30

        elif model_type == "logistic_regression_legacy":
            vectorizer, classifier = classifier_data[1], classifier_data[2]
            X = vectorizer.transform([title])
            proba = classifier.predict_proba(X)[0]
            return proba[2] * 60 + proba[1] * 30

        elif model_type == "sentence_transformer":
            st_model, classifier = classifier_data[1], classifier_data[2]
            embedding = st_model.encode([title])
            proba = classifier.predict_proba(embedding)[0]
            return proba[2] * 60 + proba[1] * 30

    except Exception as e:
        log.warning(f"ML scoring failed for '{title}': {e}")
        return 0

def rank_articles(articles, classifier_data):
    scored = []

    for article in articles:
        title = article.get("title", "")
        if not title:
            continue

        ml_score = get_ml_score(title, classifier_data)
        heuristic_score = calculate_heuristic_score(article)
        total_score = ml_score + heuristic_score

        article["ml_score"] = round(ml_score, 2)
        article["heuristic_score"] = round(heuristic_score, 2)
        article["total_score"] = round(total_score, 2)
        article["category"] = get_topic_category(article)
        scored.append(article)

    scored.sort(key=lambda x: x["total_score"], reverse=True)
    return scored

def update_priorities_in_db(articles):
    connection = sqlite3.connect("pipeline.db")
    cursor = connection.cursor()
    for article in articles:
        score = article.get("total_score", 0)
        if score >= 40:
            priority = "High"
        elif score >= 20:
            priority = "Medium"
        else:
            priority = "Low"
        cursor.execute("""
            UPDATE articles SET priority = ? WHERE url = ?
        """, (priority, article.get("url")))
    connection.commit()
    connection.close()
    log.info(f"Updated priorities in database for {len(articles)} articles")

def main():
    start_time = datetime.now()
    log.info("Starting ranker pipeline...")

    try:
        with open("raw_news.json", "r", encoding="utf-8") as f:
            articles = json.load(f)
        log.info(f"Loaded {len(articles)} articles from raw_news.json")
    except FileNotFoundError:
        log.error("raw_news.json not found. Run ingest.py first.")
        return

    titles, labels = load_labeled_data()
    
    # Try loading existing classifier first
    classifier_data = load_classifier()
    
    # If no classifier exists, train one
    if classifier_data is None:
        classifier_data = train_classifier(titles, labels)

    ranked = rank_articles(articles, classifier_data)

    # ─── TOPIC DIVERSITY — max 3 per category ───────────────────
    seen_categories = {}
    diverse_articles = []
    for article in ranked:
        category = article.get("category", "other")
        seen_categories[category] = seen_categories.get(category, 0) + 1
        if seen_categories[category] <= 3:
            diverse_articles.append(article)
        if len(diverse_articles) >= 20:
            break

    log.info(f"Top 20 after diversity filter:")
    categories_used = {}
    for a in diverse_articles:
        cat = a.get("category", "other")
        categories_used[cat] = categories_used.get(cat, 0) + 1
    for cat, count in categories_used.items():
        log.info(f"  {cat}: {count} articles")

    with open("ranked_news.json", "w", encoding="utf-8") as f:
        json.dump(diverse_articles, f, indent=2, ensure_ascii=False)

    elapsed = (datetime.now() - start_time).total_seconds()
    log.info(f"Ranked {len(ranked)} articles in {elapsed:.2f} seconds")
    log.info(f"Top 20 (diverse) saved to ranked_news.json")

    update_priorities_in_db(ranked)

    print("\n── TOP 5 ARTICLES ──────────────────────────────")
    for i, a in enumerate(diverse_articles[:5], 1):
        print(f"{i}. [{a['total_score']:.0f}] [{a['category']}] {a['title']}")
    print("────────────────────────────────────────────────\n")

if __name__ == "__main__":
    main()