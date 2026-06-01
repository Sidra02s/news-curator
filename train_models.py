"""
train_models.py — Trains and compares 3 models on labeled_headlines.csv:
1. Logistic Regression + TF-IDF (current baseline)
2. Naive Bayes + TF-IDF
3. Sentence Transformer + Logistic Regression (best quality)

Saves the best model as classifier.pkl and prints a comparison report.
Usage: python train_models.py
"""

import csv
import re
import pickle
import json
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def load_labeled_data(filepath="labeled_headlines.csv"):
    titles, labels = [], []
    label_map = {"high": 2, "medium": 1, "low": 0}

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            title = row[0].strip().strip('"').strip("'")
            priority = row[-1].strip().lower()
            priority = re.sub(r'[^a-z]', '', priority)
            if priority in label_map and len(title) > 5:
                titles.append(title)
                labels.append(label_map[priority])

    log.info(f"Loaded {len(titles)} labeled headlines")
    dist = Counter(labels)
    log.info(f"Distribution — Low: {dist[0]}, Medium: {dist[1]}, High: {dist[2]}")
    return titles, labels


def evaluate_model(name, pipeline, X, y, cv=5):
    """Run stratified k-fold cross validation and return mean accuracy."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy')
    log.info(f"{name}: CV Accuracy = {scores.mean():.3f} ± {scores.std():.3f}")
    return scores.mean(), scores.std()


def train_logistic_tfidf(titles, labels):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english')
    classifier = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    pipeline = Pipeline([('tfidf', vectorizer), ('clf', classifier)])
    mean, std = evaluate_model("Logistic Regression + TF-IDF", pipeline, titles, labels)
    pipeline.fit(titles, labels)
    return pipeline, mean, std


def train_naive_bayes(titles, labels):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english')
    classifier = MultinomialNB(alpha=0.1)
    pipeline = Pipeline([('tfidf', vectorizer), ('clf', classifier)])
    mean, std = evaluate_model("Naive Bayes + TF-IDF", pipeline, titles, labels)
    pipeline.fit(titles, labels)
    return pipeline, mean, std


def train_sentence_transformer(titles, labels):
    """Train Logistic Regression on sentence transformer embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
        log.info("Loading sentence transformer model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        log.info("Encoding titles...")
        embeddings = model.encode(titles, show_progress_bar=False)
        
        classifier = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(classifier, embeddings, labels, cv=skf, scoring='accuracy')
        mean, std = scores.mean(), scores.std()
        log.info(f"Sentence Transformer + LR: CV Accuracy = {mean:.3f} ± {std:.3f}")
        
        classifier.fit(embeddings, labels)
        return model, classifier, mean, std
        
    except ImportError:
        log.warning("sentence-transformers not installed. Run: pip install sentence-transformers")
        return None, None, 0, 0


def save_best_model(results, lr_pipeline, nb_pipeline, st_model, st_classifier):
    """Save the best performing model as classifier.pkl."""
    best_name = max(results, key=lambda x: results[x][0])
    best_acc = results[best_name][0]
    
    log.info(f"\n{'='*50}")
    log.info("MODEL COMPARISON RESULTS:")
    for name, (mean, std) in results.items():
        marker = " ← BEST" if name == best_name else ""
        log.info(f"  {name}: {mean:.3f} ± {std:.3f}{marker}")
    log.info(f"{'='*50}")
    
    if best_name == "Sentence Transformer + LR" and st_model is not None:
        with open("classifier.pkl", "wb") as f:
            pickle.dump(("sentence_transformer", st_model, st_classifier), f)
        log.info("Saved Sentence Transformer model as classifier.pkl")
    elif best_name == "Naive Bayes + TF-IDF":
        with open("classifier.pkl", "wb") as f:
            pickle.dump(("naive_bayes", nb_pipeline), f)
        log.info("Saved Naive Bayes model as classifier.pkl")
    else:
        with open("classifier.pkl", "wb") as f:
            pickle.dump(("logistic_regression", lr_pipeline), f)
        log.info("Saved Logistic Regression model as classifier.pkl")

    # Save comparison report as JSON for the metrics dashboard
    report = {
        "models": {name: {"accuracy": round(mean, 4), "std": round(std, 4)} 
                   for name, (mean, std) in results.items()},
        "best_model": best_name,
        "best_accuracy": round(best_acc, 4)
    }
    with open("model_comparison.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info("Saved model comparison to model_comparison.json")
    
    return best_name, best_acc


def main():
    titles, labels = load_labeled_data()

    if len(titles) < 50:
        log.error("Not enough data. Run autolabel.py first.")
        return

    results = {}

    log.info("\nTraining Model 1: Logistic Regression + TF-IDF")
    lr_pipeline, lr_mean, lr_std = train_logistic_tfidf(titles, labels)
    results["Logistic Regression + TF-IDF"] = (lr_mean, lr_std)

    log.info("\nTraining Model 2: Naive Bayes + TF-IDF")
    nb_pipeline, nb_mean, nb_std = train_naive_bayes(titles, labels)
    results["Naive Bayes + TF-IDF"] = (nb_mean, nb_std)

    log.info("\nTraining Model 3: Sentence Transformer + LR")
    st_model, st_classifier, st_mean, st_std = train_sentence_transformer(titles, labels)
    if st_model is not None:
        results["Sentence Transformer + LR"] = (st_mean, st_std)

    best_name, best_acc = save_best_model(results, lr_pipeline, nb_pipeline, st_model, st_classifier)
    log.info(f"\nBest model: {best_name} with {best_acc:.1%} accuracy")
    log.info("classifier.pkl updated. Restart ranker.py to use new model.")


if __name__ == "__main__":
    main()