"""
train_models.py — Trains and compares 3 models on labeled_headlines.csv:
1. Logistic Regression + TF-IDF (current baseline)
2. Naive Bayes + TF-IDF
3. Sentence Transformer + Logistic Regression (best quality)

Saves the best model as classifier.pkl, prints full evaluation report,
and saves confusion matrix as docs/confusion_matrix.png.
Usage: python train_models.py
"""

import csv
import re
import pickle
import json
import os
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay, f1_score
)
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

CLASS_NAMES = ["Skip", "Keep"]


def load_labeled_data(filepath="labeled_headlines.csv"):
    # Updated to capture the priority numeric scores alongside titles and labels
    titles, labels, raw_scores = [], [], []
    label_map = {
    "high": 1,
    "medium": 0,
    "low": 0,
    "keep": 1,
    "skip": 0 
    }

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Look for total_score index dynamically, otherwise assume index 1 or default to neutral score
        score_idx = next((i for i, col in enumerate(header) if 'score' in col.lower()), 1)
        
        for row in reader:
            if len(row) < 2:
                continue
            title = row[0].strip().strip('"').strip("'")
            priority = row[-1].strip().lower()
            priority = re.sub(r'[^a-z]', '', priority)
            
            # Extract numerical score for feature anchoring
            try:
                score_val = float(row[score_idx])
            except (ValueError, IndexError):
                score_val = 50.0  # Neutral fallback anchor if missing
                
            if priority in label_map and len(title) > 5:
                titles.append(title)
                labels.append(label_map[priority])
                raw_scores.append(score_val)

    log.info(f"Loaded {len(titles)} labeled headlines")
    dist = Counter(labels)
    log.info(f"Distribution — Skip: {dist[0]}, Keep: {dist[1]}")
    
    # Min-max normalization so the raw scores don't overpower the model weight gradients
    scores_array = np.array(raw_scores, dtype=np.float32).reshape(-1, 1)
    max_val = np.max(scores_array) if np.max(scores_array) > 0 else 1.0
    normalized_scores = scores_array / max_val
    
    return titles, labels, normalized_scores


def evaluate_model(name, pipeline, X, y, cv=5):
    """Run stratified k-fold cross validation and return mean accuracy."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring='f1')
    log.info(f"{name}: CV F1 = {scores.mean():.3f} ± {scores.std():.3f}")
    return scores.mean(), scores.std()


def print_full_report(name, y_test, y_pred):
    """Print per-class precision, recall, F1 and confusion matrix."""
    log.info(f"\n{'='*50}")
    log.info(f"FULL EVALUATION REPORT — {name}")
    log.info(f"{'='*50}")
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    log.info(f"\n{report}")

    cm = confusion_matrix(y_test, y_pred)
    log.info(f"Confusion Matrix (rows=actual, cols=predicted):")
    log.info(f"Classes: {CLASS_NAMES}")
    log.info(f"\n{cm}")

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    log.info(f"\nMacro F1:    {macro_f1:.3f}")
    log.info(f"Weighted F1: {weighted_f1:.3f}")

    return cm, report, macro_f1


def save_confusion_matrix_image(cm, model_name, filename="docs/confusion_matrix.png"):
    """Save confusion matrix as a PNG image."""
    os.makedirs("docs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    log.info(f"Confusion matrix saved to {filename}")


def train_logistic_tfidf(titles, labels):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english')
    classifier = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    pipeline = Pipeline([('tfidf', vectorizer), ('clf', classifier)])
    mean, std = evaluate_model("Logistic Regression + TF-IDF", pipeline, titles, labels)

    # Final fit + held-out eval for confusion matrix
    X_train, X_test, y_train, y_test = train_test_split(
        titles, labels, test_size=0.2, random_state=42, stratify=labels
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    cm, report, macro_f1 = print_full_report("Logistic Regression + TF-IDF", y_test, y_pred)

    # Refit on full data
    pipeline.fit(titles, labels)
    return pipeline, mean, std, cm, macro_f1


def train_naive_bayes(titles, labels):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english')
    classifier = MultinomialNB(alpha=0.1)
    pipeline = Pipeline([('tfidf', vectorizer), ('clf', classifier)])
    mean, std = evaluate_model("Naive Bayes + TF-IDF", pipeline, titles, labels)

    X_train, X_test, y_train, y_test = train_test_split(
        titles, labels, test_size=0.2, random_state=42, stratify=labels
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    cm, report, macro_f1 = print_full_report("Naive Bayes + TF-IDF", y_test, y_pred)

    pipeline.fit(titles, labels)
    return pipeline, mean, std, cm, macro_f1


def train_sentence_transformer(titles, labels, numerical_scores):
    """Train Logistic Regression on sentence transformer embeddings combined with numerical score markers."""
    try:
        from sentence_transformers import SentenceTransformer
        log.info("Loading sentence transformer model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        log.info("Encoding titles...")
        text_embeddings = model.encode(titles, show_progress_bar=False)

        # Horizontally merge semantic text vector with your physical priority ranking score
        X_combined = np.hstack([text_embeddings, numerical_scores])

        classifier = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(classifier, X_combined, labels, cv=skf, scoring='f1')
        mean, std = scores.mean(), scores.std()
        log.info(f"Sentence Transformer + LR: CV Accuracy = {mean:.3f} ± {std:.3f}")

        # Held-out eval for confusion matrix
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, labels, test_size=0.2, random_state=42, stratify=labels
        )
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm, report, macro_f1 = print_full_report("Sentence Transformer + LR", y_test, y_pred)

        # Refit on full data
        classifier.fit(X_combined, labels)
        return model, classifier, mean, std, cm, macro_f1

    except ImportError:
        log.warning("sentence-transformers not installed. Run: pip install sentence-transformers")
        return None, None, 0, 0, None, 0


def save_best_model(results, lr_pipeline, nb_pipeline, st_model, st_classifier, cms):
    """Save the best performing model as classifier.pkl."""
    best_name = max(results, key=lambda x: results[x][2])
    best_acc = results[best_name][2]

    log.info(f"\n{'='*50}")
    log.info("MODEL COMPARISON RESULTS:")
    for name, (mean, std, macro_f1) in results.items():
        marker = " ← BEST" if name == best_name else ""
        log.info(f"  {name}: Accuracy={mean:.3f}±{std:.3f}  MacroF1={macro_f1:.3f}{marker}")
    log.info(f"{'='*50}")

    # Save confusion matrix for best model
    if best_name in cms and cms[best_name] is not None:
        save_confusion_matrix_image(cms[best_name], best_name)

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

    report = {
        "models": {
            name: {
                "accuracy": round(mean, 4),
                "std": round(std, 4),
                "macro_f1": round(macro_f1, 4)
            }
            for name, (mean, std, macro_f1) in results.items()
        },
        "best_model": best_name,
        "best_macro_f1": round(best_acc, 4)
    }
    with open("model_comparison.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info("Saved model comparison to model_comparison.json")

    return best_name, best_acc


def main():
    # Adjusted assignment to capture the new normalized score matrix array
    titles, labels, numerical_scores = load_labeled_data()

    if len(titles) < 50:
        log.error("Not enough data. Run autolabel.py first.")
        return

    results = {}
    cms = {}

    log.info("\nTraining Model 1: Logistic Regression + TF-IDF")
    lr_pipeline, lr_mean, lr_std, lr_cm, lr_f1 = train_logistic_tfidf(titles, labels)
    results["Logistic Regression + TF-IDF"] = (lr_mean, lr_std, lr_f1)
    cms["Logistic Regression + TF-IDF"] = lr_cm

    log.info("\nTraining Model 2: Naive Bayes + TF-IDF")
    nb_pipeline, nb_mean, nb_std, nb_cm, nb_f1 = train_naive_bayes(titles, labels)
    results["Naive Bayes + TF-IDF"] = (nb_mean, nb_std, nb_f1)
    cms["Naive Bayes + TF-IDF"] = nb_cm

    log.info("\nTraining Model 3: Sentence Transformer + LR")
    # Passed numerical scores directly down into the embedding feature space
    st_model, st_classifier, st_mean, st_std, st_cm, st_f1 = train_sentence_transformer(titles, labels, numerical_scores)
    if st_model is not None:
        results["Sentence Transformer + LR"] = (st_mean, st_std, st_f1)
        cms["Sentence Transformer + LR"] = st_cm

    best_name, best_acc = save_best_model(
        results, lr_pipeline, nb_pipeline, st_model, st_classifier, cms
    )
    log.info(f"\nBest model: {best_name} with {best_acc:.1%} accuracy")
    log.info("classifier.pkl updated. Restart ranker.py to use new model.")


if __name__ == "__main__":
    main()