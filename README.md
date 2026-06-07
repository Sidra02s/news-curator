# 📰 AI News Curator

A fully automated, self-improving personal news pipeline that fetches, ranks, summarizes, and delivers a daily morning briefing to Telegram — powered by machine learning.

## Demo

> Every morning at 6am UAE time, this pipeline runs automatically and sends a briefing like this:

```
SIDRA'S MORNING BRIEFING
Sunday, June 07, 2026
-----------------------------

🌍 World & Politics
• Peru's voters face a straight left-right choice in their election runoff. [Read more]
• Ukraine unleashed a drone swarm on Russia after Putin rejected Zelenskyy talks. [Read more]

🤖 Tech & AI
• OpenAI rolls out Lockdown Mode for ChatGPT to protect sensitive data
  from prompt injection attacks. [Read more]

💄 Fashion & Beauty
• The Rocky Horror Show is back on Broadway — Allure got the scoop on its
  glittery, out-of-this-world hair and makeup. [Read more]

💪 Health & Wellness
• GLP-1 drugs like Ozempic may significantly cut breast cancer risk — a
  potential game-changer for women's health. [Read more]
• A new pill nearly doubles survival rates for pancreatic cancer patients
  in trials. [Read more]
• Waist-to-hip ratio is emerging as a better obesity indicator than BMI. [Read more]

🇮🇳 India
• Government defends LPG price hikes, claiming India's rates are still
  among the world's lowest despite a 46% global jump. [Read more]
• Supreme Court backs curbs on online gaming in major GST levy ruling. [Read more]

TODAY'S TAKEAWAY: New medical breakthroughs are giving real hope, while
the rest of the world keeps doing its usual chaotic thing.
-----------------------------
```

---

## Live

[View today's briefing →](https://Sidra02s.github.io/news-curator)

## Architecture

```
ingest.py         → Fetch ~190 articles from NewsAPI, Guardian, 13 RSS feeds
     ↓
ranker.py         → Score with Sentence Transformer + heuristic model
     ↓
summarizer.py     → Generate briefing via Gemini 2.5 Flash
     ↓
delivery.py       → Send to Telegram with 👍👎 feedback buttons
     ↓
feedback_listener → Capture votes → SQLite DB
     ↓
Sunday retrain    → feedback_to_csv + autolabel + train_models → new classifier.pkl
```

---

## Features

- **Multi-source ingestion** — NewsAPI, The Guardian API, 13 RSS feeds across India, Tech/AI, Health, Fashion, Beauty, Music
- **Fuzzy deduplication** — RapidFuzz catches near-duplicate headlines across sources
- **ML ranking** — Sentence Transformer (all-MiniLM-L6-v2) embeddings combined with normalized heuristic scores fed into Logistic Regression
- **3-model comparison** — Logistic Regression + TF-IDF, Naive Bayes + TF-IDF, Sentence Transformer + LR evaluated with F1, precision, recall, and confusion matrix
- **Topic diversity filter** — max 5 articles per category, top 20 overall
- **LLM summarization** — Gemini 2.5 Flash generates a sharp, opinionated briefing
- **Feedback loop** — 👍👎 Telegram buttons → SQLite → weekly retraining → better rankings
- **Automated CI/CD** — GitHub Actions runs the full pipeline daily at 2am UTC, retrains every Sunday
- **Metrics dashboard** — per-run stats (articles ingested, accuracy, runtime) tracked in SQLite and rendered as HTML

---

## ML Pipeline

The classifier is trained on labeled news headlines using binary Keep/Skip classification.

### Models Compared

| Model | CV F1 |
|---|---|
| Logistic Regression + TF-IDF | ~0.50 |
| Naive Bayes + TF-IDF | ~0.48 |
| **Sentence Transformer + LR** | **~0.57** |

The winning model combines 384-dimensional semantic embeddings from `all-MiniLM-L6-v2` with a normalized heuristic score (recency, source trust, keyword signals) as an additional feature — making training and inference consistent.

### Self-Improvement Loop

```
Daily use → Telegram feedback buttons → votes saved to DB
                                              ↓
                              Sunday: feedback_to_csv.py converts votes to labels
                                              ↓
                                       autolabel.py labels new articles via Groq
                                              ↓
                                    train_models.py retrains classifier
                                              ↓
                               New classifier.pkl committed to repo
                                              ↓
                              Next morning: better rankings
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML | Sentence Transformers, scikit-learn |
| LLM — Summarization | Gemini 2.5 Flash |
| LLM — Autolabeling | Groq (LLaMA 3.3 70B) |
| Database | SQLite |
| Delivery | python-telegram-bot |
| CI/CD | GitHub Actions |
| Deduplication | RapidFuzz |

---

## Project Structure

```
news-curator/
├── ingest.py              # Fetch and deduplicate articles
├── ranker.py              # ML + heuristic scoring
├── summarizer.py          # Gemini briefing generation
├── delivery.py            # Telegram delivery with feedback buttons
├── feedback_listener.py   # Capture Telegram votes to DB
├── feedback_to_csv.py     # Convert votes to training labels
├── autolabel.py           # Groq-powered auto-labeling
├── train_models.py        # 3-model comparison + retraining
├── main.py                # Pipeline orchestrator
├── init_db.py             # Database schema initialization
├── metrics.py             # Dashboard generation
├── labeled_headlines.csv  # Training dataset (~700 labeled headlines)
├── classifier.pkl         # Trained model (Sentence Transformer + LR)
├── .github/workflows/
│   └── briefing.yml       # Daily + weekly GitHub Actions jobs
└── docs/
    ├── briefing.txt        # Latest briefing
    ├── dashboard.html      # Metrics dashboard
    └── confusion_matrix.png
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Sidra02s/news-curator.git
cd news-curator
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file:

```
NEWS_API_KEY=your_newsapi_key
GUARDIAN_API_KEY=your_guardian_key
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 3. Initialize database

```bash
python init_db.py
```

### 4. Run the pipeline

```bash
python main.py
```

### 5. Capture feedback (run after delivery)

```bash
python feedback_listener.py
```

### 6. Retrain manually

```bash
python autolabel.py
python train_models.py
```

---

## GitHub Actions

Two automated jobs:

**Daily (2am UTC / 6am UAE):**
Runs the full pipeline — ingest → rank → summarize → deliver → metrics

**Weekly (Sundays 3am UTC):**
Merges feedback votes → autolabels new articles → retrains classifier → commits updated model

---

## API Keys Required

| Service | Purpose | Free Tier |
|---|---|---|
| NewsAPI | Article fetching | 100 req/day |
| The Guardian | Article fetching | 500 req/day |
| Gemini 2.5 Flash | Briefing summarization | 20 req/day |
| Groq (LLaMA 3.3 70B) | Auto-labeling | 1000 req/day |
| Telegram Bot API | Delivery | Free |

---

## Model Design Decision

The classifier is intentionally tuned to favor **high recall on Keep-priority news**. In a personal news assistant, missing a major story is far worse than surfacing a borderline one. So the model is optimized to minimize false negatives — it would rather show you one extra article you didn't need than bury something important.

This is a deliberate product decision, not a model weakness.

## What I'd Improve Next

- Replace SQLite with PostgreSQL for persistent feedback across GitHub Actions runs
- Fine-tune the embedding model on domain-specific news data
- Add more labeled data to push F1 above 0.70
- Expand to more topics and RSS sources