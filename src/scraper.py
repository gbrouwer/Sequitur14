# === FILE: scrapers.py ===
# === PURPOSE: Fetch documents from arXiv based on time intervals and category ===

import feedparser
import time
import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def scrape_arxiv(config):
    base_url = "http://export.arxiv.org/api/query"
    category = config["category"]
    start_year = config["start_year"]
    end_year = config["end_year"]
    max_results = config.get("max_results_per_month", 100)
    sampling_freq = config.get("sampling_freq", "monthly")
    raw_dir = Path("../data") / config["data_name"] / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for start, end in _generate_time_ranges(start_year, end_year, sampling_freq):
        filename = f"raw_{start.date()}_to_{end.date()}.csv"
        filepath = raw_dir / filename

        if filepath.exists():
            print(f"✅ Skipping {filename} (already exists)")
            continue

        print(f"⏳ Querying {start.date()} to {end.date()}...")
        entries = _scrape_range(base_url, category, start, end, max_results)

        if entries:
            _save_entries(entries, filepath)
        else:
            print(f"⚠️  No results for {filename}")

        time.sleep(3)  # polite delay

def _scrape_range(base_url, category, start_date, end_date, max_results):
    start_str = start_date.strftime("%Y%m%d%H%M")
    end_str = end_date.strftime("%Y%m%d%H%M")

    query = (
        f"search_query=cat:{category}+AND+submittedDate:[{start_str}+TO+{end_str}]"
        f"&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    )
    url = f"{base_url}?{query}"
    feed = feedparser.parse(url)
    return feed.entries if feed.entries else []

def _save_entries(entries, path):
    records = []
    for entry in entries:
        records.append({
            "id": entry.get("id"),
            "title": entry.get("title"),
            "summary": entry.get("summary"),
            "published": entry.get("published"),
            "authors": ", ".join(author.name for author in entry.get("authors", [])),
            "link": entry.get("link")
        })
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)

def _generate_time_ranges(start_year, end_year, sampling_freq):
    current = datetime(start_year, 1, 1)
    end = datetime(end_year + 1, 1, 1)

    if sampling_freq == "monthly":
        def next_step(d):
            year, month = d.year + int(d.month == 12), (d.month % 12) + 1
            return datetime(year, month, 1)
    elif sampling_freq == "weekly":
        def next_step(d):
            return d + timedelta(weeks=1)
    else:
        def next_step(d):
            return d + timedelta(days=1)

    while current < end:
        next_time = next_step(current)
        yield current, next_time
        current = next_time
