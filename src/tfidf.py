# === FILE: run_tfidf_timesliced.py ===
# === PURPOSE: Compute TF-IDF keywords using rolling time windows and full dataset ===

import pandas as pd
from datetime import timedelta
from sequitur14.managers import JobManager
from sequitur14.analyzers import TfidfAnalyzer
from utils import Utils


def run_tfidf_timesliced(config):
    """
    Run TF-IDF keyword extraction over rolling time windows.
    Each window spans a fixed number of days and slides forward by a given step size.
    Also runs TF-IDF analysis on the full corpus at the end.
    """
    job = JobManager(config=config, mode="analyze")
    job.print_header()

    # Load cleaned corpus and metadata
    corpus = Utils.load_corpus(config["data_name"])
    metadata = Utils.load_metadata(config["data_name"])
    metadata["published"] = pd.to_datetime(metadata["published"], errors="coerce")

    # Load stopwords if provided
    stopwords = set()
    if config.get("stopword_source"):
        with open(config["stopword_source"], "r", encoding="utf-8") as f:
            stopwords = set(w.strip() for w in f if w.strip())

    window_size = config.get("rolling_window_days", 30)
    step_size = config.get("rolling_step_days", 7)
    min_docs = config.get("min_docs_per_window", 10)

    start_date = metadata["published"].min()
    end_date = metadata["published"].max()

    analyzer = TfidfAnalyzer(config)

    current_start = start_date
    while current_start + timedelta(days=window_size) <= end_date:
        current_end = current_start + timedelta(days=window_size)
        mask = (metadata["published"] >= current_start) & (metadata["published"] < current_end)
        idxs = metadata[mask].index.tolist()

        timestamp_str = current_start.strftime("%Y-%m-%d")

        print(f"\nAnalyzing window {current_start.date()} to {current_end.date()} ({len(idxs)} docs)")

        if len(idxs) < min_docs:
            print("Skipping: not enough documents.")
            current_start += timedelta(days=step_size)
            continue

        docs = [corpus[i] for i in idxs]
        keywords = analyzer.fit(docs)

        # Remove keywords that are in the stopword list
        if stopwords:
            keywords = [kw for kw in keywords if kw[0] not in stopwords]

        output_path = job.results_dir / f"keywords_{timestamp_str}.csv"
        Utils.save_keywords(keywords, output_path)

        current_start += timedelta(days=step_size)

    # Run full-corpus TF-IDF at the end
    print("\nAnalyzing full corpus...")
    keywords = analyzer.fit(corpus)
    if stopwords:
        keywords = [kw for kw in keywords if kw[0] not in stopwords]

    full_path = job.results_dir / "keywords_full_corpus.csv"
    Utils.save_keywords(keywords, full_path)

    print("\nTF-IDF rolling window and full corpus analysis complete.")
