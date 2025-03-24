# === FILE: lda.py ===
# === PURPOSE: Compute LDA topics for each time slice (monthly or weekly) ===

import pandas as pd
from datetime import datetime
from sequitur14.managers import JobManager
from sequitur14.analyzers import LdaAnalyzer
from utils import save_topics, save_document_topics, load_corpus, load_metadata


def run_lda_timesliced(config):
    """
    Run LDA topic modeling on corpus grouped by time slices.
    Saves topic-word matrices and document-topic distributions per slice.
    """
    job = JobManager(config=config, base_results_path="../results/lda", mode="analysis")
    job.print_header()

    # Load cleaned corpus and metadata
    corpus = load_corpus(config["data_name"])
    metadata = load_metadata(config["data_name"])
    metadata["published"] = pd.to_datetime(metadata["published"], errors="coerce")

    # Create time slice labels
    freq = config.get("sampling_freq", "monthly")
    if freq == "weekly":
        metadata["timeslice"] = metadata["published"].dt.to_period("W-MON").dt.start_time
    else:
        metadata["timeslice"] = metadata["published"].dt.to_period("M").dt.to_timestamp()

    # Group indices by time slice
    groups = metadata.groupby("timeslice").indices

    for ts, idxs in sorted(groups.items()):
        timestamp_str = ts.strftime("%Y-%m") if freq == "monthly" else ts.strftime("%Y-%m-%d")
        print(f"\n🔍 Running LDA for {timestamp_str} ({len(idxs)} docs)")

        docs = [corpus[i] for i in idxs]
        analyzer = LdaAnalyzer(
            corpus=docs,
            n_topics=config.get("lda_n_topics", 10),
            max_features=config.get("lda_max_features", 1000),
            stop_words=config.get("remove_top_n_stopwords"),
            stopword_path=config.get("stopword_source")
        )
        analyzer.fit()

        topic_words = analyzer.topics_df
        doc_topics = analyzer.document_topics_df

        topics_path = job.base_dir / f"topics_{timestamp_str}.csv"
        doc_topics_path = job.base_dir / f"doc_topics_{timestamp_str}.csv"

        save_topics(topic_words, topics_path)
        save_document_topics(doc_topics, doc_topics_path)

    print("\n✅ Time-sliced LDA analysis complete.")
