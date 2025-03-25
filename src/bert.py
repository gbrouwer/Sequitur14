# === FILE: bert.py ===
# === PURPOSE: Run full + time-sliced BERT embedding + BERTopic clustering ===

import pandas as pd
from pathlib import Path
from sequitur14.managers import JobManager
from sequitur14.embedders import BertEmbedder
from sequitur14.clusterers import BertClusterer
from sequitur14.exporters import ResultsExporter
from utils import load_corpus
from utils import load_metadata
from datetime import datetime
import json


def run_bert_timesliced(config):
    """
    Run BERTopic analysis on full corpus and across rolling time slices.

    Parameters:
        config_path (str): Path to the job config.yaml file
    """

    job = JobManager(config, mode="bert")
    if not job.check_pipeline_ready():
        raise RuntimeError("Pipeline is not ready. Make sure scraping, staging, and preprocessing are complete.")

    out_dir = job.get_results_path("bert")
    job.print_header()

    # Load corpus and metadata
    corpus = load_corpus(config["job_name"])
    metadata = load_metadata(config["job_name"])
    metadata["published"] = pd.to_datetime(metadata["published"], errors="coerce")


    # === Embed full corpus ===
    embedder = BertEmbedder(
        model_path=config.get("model_path", "all-MiniLM-L6-v2"),
        device=config.get("device", "cpu")
    )
    embeddings = embedder.encode(corpus)

    # === Fit BERTopic on full corpus ===
    clusterer = BertClusterer(
        model_path=config.get("model_path", "all-MiniLM-L6-v2"),
        reduce=config.get("reduce", True),
        n_topics=config.get("n_topics", "auto")
    )
    clusterer.fit(corpus, embeddings)

    out_dir = job.get_results_path("bert")
    clusterer.df_docs.to_csv(out_dir / "doc_topics_full_corpus.csv", index=False)

    # Save full topic words
    topic_words = []
    for topic_id in clusterer.topic_model.get_topic_info()["Topic"].tolist():
        if topic_id == -1:
            continue
        words = clusterer.topic_model.get_topic(topic_id)
        for word, score in words:
            topic_words.append({"topic": topic_id, "keyword": word, "score": score})
    pd.DataFrame(topic_words).to_csv(out_dir / "topics_full_corpus.csv", index=False)

    print("\tFull corpus BERTopic completed.")

    # === Rolling window analysis ===
    print("\t[Step 2] Rolling BERTopic Inference:")
    combined_slices = []
    slices = job.get_rolling_windows(metadata)

    for timestamp_str, df_slice in slices:
        docs = df_slice["title"] + " " + df_slice["summary"]
        slice_embeddings = embedder.encode(docs.tolist())

        df = clusterer.transform(docs.tolist(), slice_embeddings)
        df["timestamp"] = timestamp_str
        combined_slices.append(df)

        # Save per-slice document topics
        df.to_csv(out_dir / f"doc_topics_{timestamp_str}.csv", index=False)

        # Save topic keywords for this slice
        topic_words = []
        for topic_id in clusterer.topic_model.get_topic_info()["Topic"].tolist():
            if topic_id == -1:
                continue
            words = clusterer.topic_model.get_topic(topic_id)
            for word, score in words:
                topic_words.append({"topic": topic_id, "keyword": word, "score": score})
        pd.DataFrame(topic_words).to_csv(out_dir / f"topics_{timestamp_str}.csv", index=False)

    # Save combined rolling output
    all_df = pd.concat(combined_slices)
    all_df.to_csv(out_dir / "doc_topics_rolling_all.csv", index=False)

    print("\tTime-sliced BERTopic inference completed.")

    # === Export to visualization-ready folder ===
    job.snapshot_to_results()
    combined_json_path = out_dir / "doc_topics_animation.json"
    export_animated_topicmap(all_df, combined_json_path)

def export_animated_topicmap(df, out_path):
    """
    Convert time-sliced doc-topic data into a D3-friendly JSON structure.
    Each frame contains (x, y, topic) for all points in that timestamp.
    
    Parameters:
        df (pd.DataFrame): Combined doc_topics_rolling_all.csv
        out_path (Path): Output JSON path
    """
    grouped = df.groupby("timestamp")
    frames = []

    for timestamp, group in grouped:
        frame = {
            "timestamp": timestamp,
            "points": [
                {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "topic": int(row["topic"])
                }
                for _, row in group.iterrows()
                if pd.notnull(row["x"]) and pd.notnull(row["y"])
            ]
        }
        frames.append(frame)

    with open(out_path, "w") as f:
        json.dump(frames, f)

    print(f"\tD3 animation JSON saved to: {out_path}")


"""
run_bert_timesliced(): Run full + rolling BERTopic analysis on a preprocessed corpus.

This function embeds all text using a pre-trained BERT model (e.g., MiniLM) and applies
BERTopic for unsupervised topic modeling. It first fits the model on the full corpus,
then reuses that model to assign topics and 2D UMAP coordinates for each rolling slice.

Method: BERTopic uses precomputed BERT embeddings + UMAP + HDBSCAN + class-based TF-IDF.
The formula for topic extraction:
    - Embeddings: dense vector `E_d = BERT(text_d)`
    - Dim reduction: `E_d' = UMAP(E_d)`
    - Clustering: topics = HDBSCAN(E_d')
    - Keywords: scored using class-based TF-IDF:
        tfidf(t, T) = tf(t, T) * idf(t, D)

Implementation: The full model is trained once and applied to each slice using `.transform()`.
Topic assignments, probabilities, and UMAP (x, y) positions are saved per slice, with
timestamp labels. Outputs are standardized via `ResultsExporter` for downstream D3 visualization.
"""
