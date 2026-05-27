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



\## Project Structure

