import json
from pathlib import Path
import pandas as pd


def save_corpus(corpus, path):
    with open(path, "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(doc.strip() + "\n")


def load_corpus(job_name):
    path = Path("../data") / job_name / "corpus" / "corpus.txt"
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
    df = pd.DataFrame(metadata)
    df.to_parquet(path, index=False)


def load_metadata(job_name):
    path = Path("../data") / job_name / "corpus" / "metadata.parquet"
    return pd.read_parquet(path)
