# === FILE: bert.py ===
# === PURPOSE: Run BERTopic clustering for each time slice (monthly or weekly) ===

import pandas as pd
from datetime import datetime
from sequitur14.managers import JobManager
from utils import load_corpus, load_metadata, save_topics, save_document_topics
from sequitur14.embedders import BertEmbedder
from sequitur14.clusterers import BertClusterer
import numpy as np


def run_bert_timesliced(config):
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

    # Group docs by slice
    groups = metadata.groupby("timeslice").indices

    embedder = BertEmbedder(
        model_path=config["embedding_model"],
        device=config.get("device", "cpu")
    )
    clusterer = BertClusterer(
        model_path=config["embedding_model"],
        reduce=config.get("umap_reduce", True),
        n_topics=config.get("n_topics", "auto")
    )

    for ts, idxs in sorted(groups.items()):
        timestamp_str = ts.strftime("%Y-%m") if freq == "monthly" else ts.strftime("%Y-%m-%d")
        print(f"\n🔍 Running BERT clustering for {timestamp_str} ({len(idxs)} docs)")

        docs = [corpus[i] for i in idxs]
        if not docs:
            continue

        # Step 1: Embed
        embeddings = embedder.encode(docs)

        # Step 2: Cluster
        topics, doc_topics = clusterer.fit(docs, embeddings)

        # Step 3: Save
        topics_path = job.base_dir / f"topics_{timestamp_str}.csv"
        doc_topics_path = job.base_dir / f"doc_topics_{timestamp_str}.csv"

        save_topics(topics, topics_path)
        save_document_topics(doc_topics, doc_topics_path)

        # Optionally save embeddings
        if config.get("save_embeddings", False):
            emb_path = job.base_dir / f"embeddings_{timestamp_str}.npy"
            np.save(emb_path, embeddings)

    print("\n✅ Time-sliced BERT clustering complete.")