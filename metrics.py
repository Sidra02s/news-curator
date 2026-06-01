"""
metrics.py — Records pipeline run stats to SQLite and generates
a static HTML dashboard saved to docs/dashboard.html

Called at the end of main.py after every pipeline run.
"""

import sqlite3
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo


def record_run(
    articles_ingested=0,
    articles_after_dedup=0,
    duplicates_blocked=0,
    high_priority_count=0,
    medium_priority_count=0,
    low_priority_count=0,
    sources_newsapi=0,
    sources_guardian=0,
    sources_rss=0,
    classifier_accuracy=0.0,
    pipeline_runtime_seconds=0.0,
    briefing_length=0,
    briefing_generated=False
):
    """Save metrics for this pipeline run to the database."""
    try:
        conn = sqlite3.connect("pipeline.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO pipeline_runs (
                articles_ingested, articles_after_dedup, duplicates_blocked,
                high_priority_count, medium_priority_count, low_priority_count,
                sources_newsapi, sources_guardian, sources_rss,
                classifier_accuracy, pipeline_runtime_seconds,
                briefing_length, briefing_generated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            articles_ingested, articles_after_dedup, duplicates_blocked,
            high_priority_count, medium_priority_count, low_priority_count,
            sources_newsapi, sources_guardian, sources_rss,
            classifier_accuracy, pipeline_runtime_seconds,
            briefing_length, 1 if briefing_generated else 0
        ))
        conn.commit()
        conn.close()
        print("📊 Pipeline metrics saved to database")
    except Exception as e:
        print(f"Metrics save error: {e}")


def fetch_all_runs():
    """Fetch all pipeline run records from DB."""
    try:
        conn = sqlite3.connect("pipeline.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT run_at, articles_ingested, articles_after_dedup,
                   duplicates_blocked, high_priority_count, medium_priority_count,
                   low_priority_count, sources_newsapi, sources_guardian,
                   sources_rss, classifier_accuracy, pipeline_runtime_seconds,
                   briefing_length, briefing_generated
            FROM pipeline_runs
            ORDER BY run_at DESC
            LIMIT 30
        """)
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"Metrics fetch error: {e}")
        return []


def fetch_source_breakdown():
    """Get article count per source from articles table."""
    try:
        conn = sqlite3.connect("pipeline.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT source, COUNT(*) as count
            FROM articles
            WHERE source IS NOT NULL
            GROUP BY source
            ORDER BY count DESC
            LIMIT 15
        """)
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"Source breakdown error: {e}")
        return []


def fetch_priority_totals():
    """Get total articles per priority tier."""
    try:
        conn = sqlite3.connect("pipeline.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT priority, COUNT(*) as count
            FROM articles
            WHERE priority IS NOT NULL
            GROUP BY priority
        """)
        rows = cursor.fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}
    except Exception as e:
        print(f"Priority totals error: {e}")
        return {}


def load_model_comparison():
    """Load model comparison results if available."""
    try:
        with open("model_comparison.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def generate_dashboard():
    """Generate static HTML dashboard and save to docs/dashboard.html"""
    runs = fetch_all_runs()
    sources = fetch_source_breakdown()
    priority_totals = fetch_priority_totals()
    model_data = load_model_comparison()

    uae_tz = ZoneInfo("Asia/Dubai")
    now = datetime.now(uae_tz).strftime("%B %d, %Y at %H:%M UAE")

    # Build run rows for table
    run_rows = ""
    run_dates, high_counts, dedup_rates, runtimes = [], [], [], []

    for run in runs:
        (run_at, ingested, after_dedup, dupes, high, medium, low,
         newsapi, guardian, rss, accuracy, runtime, brief_len, generated) = run

        dedup_rate = round((dupes / ingested * 100), 1) if ingested > 0 else 0
        status = "✅" if generated else "❌"
        acc_str = f"{accuracy:.1%}" if accuracy else "N/A"

        run_rows += f"""
        <tr>
            <td>{run_at[:16]}</td>
            <td>{ingested}</td>
            <td>{after_dedup}</td>
            <td>{dedup_rate}%</td>
            <td>{high}</td>
            <td>{acc_str}</td>
            <td>{runtime:.1f}s</td>
            <td>{status}</td>
        </tr>"""

        run_dates.append(f'"{run_at[:10]}"')
        high_counts.append(str(high))
        dedup_rates.append(str(dedup_rate))
        runtimes.append(str(round(runtime, 1)))

    # Reverse for chronological chart order
    run_dates = list(reversed(run_dates))
    high_counts = list(reversed(high_counts))
    dedup_rates = list(reversed(dedup_rates))
    runtimes = list(reversed(runtimes))

    # Source breakdown rows
    source_rows = ""
    for source, count in sources[:10]:
        source_rows += f"<tr><td>{source}</td><td>{count}</td></tr>"

    # Priority totals
    total_high = priority_totals.get("High", 0)
    total_medium = priority_totals.get("Medium", 0)
    total_low = priority_totals.get("Low", 0)
    total_articles = total_high + total_medium + total_low

    # Model comparison section
    model_html = ""
    if model_data:
        best = model_data.get("best_model", "")
        model_html = "<h2>Model Comparison</h2><table><tr><th>Model</th><th>Accuracy</th><th>Std Dev</th><th></th></tr>"
        for name, stats in model_data.get("models", {}).items():
            badge = " 🏆" if name == best else ""
            model_html += f"<tr><td>{name}{badge}</td><td>{stats['accuracy']:.1%}</td><td>±{stats['std']:.3f}</td><td>{'Best' if name == best else ''}</td></tr>"
        model_html += "</table>"
    else:
        model_html = "<h2>Model Comparison</h2><p>Run train_models.py to see model comparison results.</p>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sidra's News Pipeline Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; padding: 24px; }}
  h1 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; }}
  h2 {{ font-size: 1.1rem; font-weight: 600; margin: 28px 0 12px; color: #aaa; text-transform: uppercase; letter-spacing: 0.05em; }}
  .subtitle {{ color: #666; font-size: 0.85rem; margin-bottom: 32px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-bottom: 32px; }}
  .card {{ background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px; padding: 20px; }}
  .card .value {{ font-size: 2rem; font-weight: 700; color: #fff; }}
  .card .label {{ font-size: 0.78rem; color: #666; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }}
  .card.green .value {{ color: #4ade80; }}
  .card.blue .value {{ color: #60a5fa; }}
  .card.purple .value {{ color: #c084fc; }}
  .card.orange .value {{ color: #fb923c; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 32px; }}
  .chart-box {{ background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px; padding: 20px; }}
  table {{ width: 100%; border-collapse: collapse; background: #1a1a1a; border-radius: 12px; overflow: hidden; margin-bottom: 32px; }}
  th {{ background: #2a2a2a; padding: 10px 14px; text-align: left; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: #888; }}
  td {{ padding: 10px 14px; border-top: 1px solid #2a2a2a; font-size: 0.875rem; }}
  tr:hover td {{ background: #222; }}
  @media (max-width: 768px) {{ .charts {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>📰 News Pipeline Dashboard</h1>
<p class="subtitle">Last updated: {now}</p>

<div class="cards">
  <div class="card blue">
    <div class="value">{total_articles}</div>
    <div class="label">Total Articles</div>
  </div>
  <div class="card green">
    <div class="value">{total_high}</div>
    <div class="label">High Priority</div>
  </div>
  <div class="card purple">
    <div class="value">{total_medium}</div>
    <div class="label">Medium Priority</div>
  </div>
  <div class="card orange">
    <div class="value">{len(runs)}</div>
    <div class="label">Pipeline Runs</div>
  </div>
</div>

<div class="charts">
  <div class="chart-box">
    <canvas id="highChart"></canvas>
  </div>
  <div class="chart-box">
    <canvas id="dedupChart"></canvas>
  </div>
</div>

<h2>Pipeline Run History</h2>
<table>
  <tr>
    <th>Run At</th>
    <th>Ingested</th>
    <th>After Dedup</th>
    <th>Dupe Rate</th>
    <th>High Priority</th>
    <th>Classifier Acc</th>
    <th>Runtime</th>
    <th>Briefing</th>
  </tr>
  {run_rows}
</table>

<h2>Top Sources</h2>
<table>
  <tr><th>Source</th><th>Articles</th></tr>
  {source_rows}
</table>

{model_html}

<script>
const dates = [{",".join(run_dates)}];
const highCounts = [{",".join(high_counts)}];
const dedupRates = [{",".join(dedup_rates)}];

new Chart(document.getElementById('highChart'), {{
  type: 'line',
  data: {{
    labels: dates,
    datasets: [{{
      label: 'High Priority Articles',
      data: highCounts,
      borderColor: '#4ade80',
      backgroundColor: 'rgba(74,222,128,0.1)',
      tension: 0.3,
      fill: true
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#aaa' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#666' }}, grid: {{ color: '#222' }} }},
      y: {{ ticks: {{ color: '#666' }}, grid: {{ color: '#222' }} }}
    }}
  }}
}});

new Chart(document.getElementById('dedupChart'), {{
  type: 'bar',
  data: {{
    labels: dates,
    datasets: [{{
      label: 'Duplicate Rate %',
      data: dedupRates,
      backgroundColor: 'rgba(96,165,250,0.6)',
      borderColor: '#60a5fa',
      borderWidth: 1
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#aaa' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#666' }}, grid: {{ color: '#222' }} }},
      y: {{ ticks: {{ color: '#666' }}, grid: {{ color: '#222' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

    os.makedirs("docs", exist_ok=True)
    with open("docs/dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("📊 Dashboard saved to docs/dashboard.html")


if __name__ == "__main__":
    generate_dashboard()