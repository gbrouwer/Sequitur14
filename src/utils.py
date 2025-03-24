# === FILE: utils.py ===
# === PURPOSE: Utility methods for corpus saving, loading, metadata, etc. ===

import json
from pathlib import Path
import pandas as pd

def save_corpus(corpus, path):
    """
    Save a list of cleaned documents as a .txt file (one per line).
    """
    with open(path, "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(doc.strip() + "\n")

def load_corpus(data_name):
    """
    Load a cleaned corpus from the processed directory.
    """
    path = Path("../data") / data_name / "processed" / "corpus.txt"
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]
    
def save_keywords(keywords, path):
    df = pd.DataFrame(keywords)
    df.to_csv(path, index=False)

def save_topics(topics, path):
    df = pd.DataFrame(topics)
    df.to_csv(path, index=False)

def save_document_topics(doc_topic_matrix, path):
    df = pd.DataFrame(doc_topic_matrix)
    df.to_csv(path, index=False)

def save_metadata(metadata, path):
    """
    Save document-level metadata (e.g., id, title, published date) to a parquet file.
    """
    df = pd.DataFrame(metadata)
    df.to_parquet(path, index=False)

def load_metadata(data_name):
    """
    Load document metadata from the processed directory.
    """
    path = Path("../data") / data_name / "processed" / "metadata.parquet"
    return pd.read_parquet(path)
