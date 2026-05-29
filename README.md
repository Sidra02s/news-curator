\# Sidra's Morning Briefing — Autonomous AI News Curator



A fully autonomous AI pipeline that fetches, ranks, and delivers a personalized morning news briefing every day at 6AM UAE time — without any manual intervention.



\## What It Does



Every morning the system automatically:

1\. Ingests news from 3 sources — NewsAPI, The Guardian API, and 13 RSS feeds

2\. Ranks articles using a custom-trained classifier + deterministic heuristics

3\. Summarizes the top stories into a personalized briefing using Gemini AI

4\. Delivers the briefing to Telegram and updates this web page



\## Architecture

ingest.py → raw\_news.json (239 articles from 3 sources)

↓

ranker.py → ranked\_news.json (top 20, topic-diverse)

↓

summarizer.py → briefing.txt (6 sections, AI-generated)

↓

delivery.py → Telegram bot

↓

GitHub Actions → runs everything at 6AM UAE time, no laptop needed

\## Why This Is Different



Most AI news tools pipe everything directly into an LLM. This system has a \*\*pre-LLM priority engine\*\* — a logistic regression classifier trained on 234 hand-labeled headlines that ranks articles before they reach the AI. The LLM only summarizes pre-filtered, high-signal content.



This means:

\- Critical stories are never buried by the model

\- Token usage is minimized

\- Output quality is more consistent



\## Tech Stack



\- \*\*Python\*\* — core pipeline

\- \*\*NewsAPI + The Guardian API + RSS\*\* — multi-source ingestion

\- \*\*scikit-learn\*\* — TF-IDF vectorizer + logistic regression classifier

\- \*\*Gemini 2.5 Flash\*\* — AI summarization

\- \*\*Telegram Bot API\*\* — delivery

\- \*\*GitHub Actions\*\* — serverless automation (cron at 02:00 UTC)



\## Topics Covered



Fashion \& Beauty · Tech \& AI · Health \& Wellness · India · World \& Politics · Music



\## Live Briefing



\[View today's briefing](https://Sidra02s.github.io/news-curator)



\## How To Run Locally



```bash

git clone https://github.com/Sidra02s/news-curator.git

cd news-curator

python -m venv venv

venv\\Scripts\\activate

pip install requests feedparser scikit-learn google-genai python-telegram-bot python-dotenv

```



Add your API keys to a `.env` file then:



```bash

python main.py

```



## Project Structure

```text
news-agent/
│
├── .github/workflows/     # GitHub Actions automation configurations (6AM UAE Cron)
│   └── main.yml           # Serverless workflow pipeline execution script
│
├── docs/                  # Public web directory for frontend hosting
│   └── index.html         # Live static dashboard rendering the morning briefing
│
├── ingest.py              # Fetches raw articles from NewsAPI, Guardian API, and 13 RSS feeds
├── ranker.py              # Runs headlines through scikit-learn Logistic Regression classifier
├── summarizer.py          # Formats top-ranked articles and calls Gemini 2.5 Flash API
├── delivery.py            # Connects to Telegram Bot API to broadcast final briefings
│
├── classifier.pkl         # Serialized ML model weights (TF-IDF Vectorizer + Classifier)
├── pipeline.db            # SQLite relational database archiving history (V2 Upgrade)
├── requirements.txt       # Production library dependency versions manifest
└── README.md              # Project documentation and engineering portfolio homepage



## 📊 Model Evaluation & Performance
The ranking layer uses a `scikit-learn` pipeline featuring a **TF-IDF Vectorizer** and a **Logistic Regression** classifier. The model evaluates incoming headlines and scores them into priority tiers.

### Classification Report
| Priority Tier | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| 🔴 **High** | 0.47 | 0.80 | **0.59** | 20 |
| 🟡 **Medium** | 1.00 | 0.10 | 0.18 | 10 |
| 🔵 **Low** | 0.30 | 0.20 | 0.24 | 15 |
| **Overall Accuracy** | | | **0.44** | **45** |

### Strategic Optimization Note
The model is intentionally tuned to favor **high recall (0.80) on High-priority news**. In the context of a personal news assistant, minimizing false negatives is critical—it is much better for the agent to surface a few extra borderline articles (lower precision) than to completely miss a major story I care about (high recall).

