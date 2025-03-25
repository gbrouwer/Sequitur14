# === FILE: extract_trend_topics.py ===
# === PURPOSE: Extract LLM-related and baseline topic trends from BERTopic output ===

import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import re

def extract_trend_topics():
    # === CONFIG ===
    RESULTS_DIR = Path("../results/bert/arxiv-ai-mvp-monthly")  # Adjust to match your actual path
    KEYWORDS = ["transformer", "llm", "language model", "gpt"]
    KEYWORD_PATTERN = re.compile("|".join(KEYWORDS), re.IGNORECASE)

    # === STEP 1: Identify LLM-related topics per month ===
    all_months = sorted([f.stem.split("_")[-1] for f in RESULTS_DIR.glob("topics_*.csv")])
    topic_map = defaultdict(dict)  # month → topic_id → label

    for month in all_months:
        topics_file = RESULTS_DIR / f"topics_{month}.csv"
        df = pd.read_csv(topics_file)
        for topic_id in df['Topic'].unique():
            topic_keywords = df[df['Topic'] == topic_id]['Keyword'].tolist()
            full_text = " ".join(topic_keywords)
            if KEYWORD_PATTERN.search(full_text):
                topic_map[month][topic_id] = "llm_related"

    # === STEP 2: Count topic appearances over time ===
    topic_presence = Counter()

    for month in all_months:
        doc_topics_file = RESULTS_DIR / f"doc_topics_{month}.csv"
        if not doc_topics_file.exists():
            continue
        doc_df = pd.read_csv(doc_topics_file)
        for topic_id in doc_df['Topic'].unique():
            topic_presence[topic_id] += 1

    # === STEP 3: Select stable topics as baseline ===
    num_months = len(all_months)
    stable_threshold = int(0.6 * num_months)
    stable_topics = [topic for topic, count in topic_presence.items() if count >= stable_threshold]

    # Remove LLM topics from stable list if overlapping
    llm_topic_ids = {tid for month_map in topic_map.values() for tid in month_map.keys()}
    baseline_topic_ids = [tid for tid in stable_topics if tid not in llm_topic_ids]

    # === STEP 4: Build trend tracking DataFrame ===
    records = []

    for month in all_months:
        doc_topics_file = RESULTS_DIR / f"doc_topics_{month}.csv"
        if not doc_topics_file.exists():
            continue
        doc_df = pd.read_csv(doc_topics_file)

        for topic_id in doc_df['Topic'].unique():
            label = "llm_related" if topic_id in topic_map.get(month, {}) else (
                "baseline" if topic_id in baseline_topic_ids else None)
            if label:
                count = (doc_df['Topic'] == topic_id).sum()
                records.append({
                    "month": month,
                    "topic_id": topic_id,
                    "label": label,
                    "doc_count": count
                })

    trend_df = pd.DataFrame(records)

    # === STEP 5: Save for D3 plotting or inspection ===
    trend_df.to_csv("../results/trend_summary_llm_vs_baseline.csv", index=False)
    print("✅ Trend summary saved to trend_summary_llm_vs_baseline.csv")
