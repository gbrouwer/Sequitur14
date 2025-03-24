# === FILE: tfidf.py ===
# === PURPOSE: Compute TF-IDF keyword rankings for each time slice (monthly or weekly) ===

import pandas as pd
from datetime import datetime
from sequitur14.managers import JobManager
from sequitur14.analyzers import TfidfAnalyzer
from utils import load_corpus, load_metadata, save_keywords


def run_tfidf_timesliced(config):
    job = JobManager(config=config, base_results_path="../results/tfidf", mode="analysis")
    job.print_header()

    # Load corpus and metadata
    corpus = load_corpus(config["data_name"])
    metadata = load_metadata(config["data_name"])
    metadata["published"] = pd.to_datetime(metadata["published"], errors="coerce")

    # Determine time slicing
    freq = config.get("sampling_freq", "monthly")
    if freq == "weekly":
        metadata["timeslice"] = metadata["published"].dt.to_period("W-MON").dt.start_time
    else:
        metadata["timeslice"] = metadata["published"].dt.to_period("M").dt.to_timestamp()

    # Group by time slice
    groups = metadata.groupby("timeslice").indices

    for ts, idxs in sorted(groups.items()):
        timestamp_str = ts.strftime("%Y-%m") if freq == "monthly" else ts.strftime("%Y-%m-%d")
        print(f"\n🔍 Running TF-IDF for {timestamp_str} ({len(idxs)} docs)")

        docs = [corpus[i] for i in idxs]
        analyzer = TfidfAnalyzer(
            corpus=docs,
            max_features=config.get("tfidf_max_features"),
            stop_words=config.get("remove_top_n_stopwords"),
            stopword_path=config.get("stopword_source")
        )
        analyzer.fit()

        keywords = analyzer.df

        filename = f"tfidf_{timestamp_str}.csv"
        out_path = job.base_dir / filename
        save_keywords(keywords, out_path)

    print("\n✅ Time-sliced TF-IDF analysis complete.")
