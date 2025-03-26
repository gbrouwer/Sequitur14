# === FILE: preprocess.py ===
# === PURPOSE: Clean and merge title + summary into a corpus ===

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sequitur14.managers import JobManager
import re
import os


def preprocess_from_staging(config):
    job = JobManager(config, mode="preprocess")
    staging_dir = job.get_data_path("staging")
    processed_dir = job.get_data_path("processed")
    corpus_dir = job.get_data_path("corpus")

    all_docs = []
    all_meta = []

    files = sorted(staging_dir.glob("arxiv_*.parquet"))

    for file in tqdm(files, desc="Preprocessing slices"):
        df = pd.read_parquet(file)

        df["title_raw"] = df["title"].apply(clean_text)
        df["summary_raw"] = df["summary"].apply(clean_text)
        df["text_raw"] = df["title_raw"] + " " + df["summary_raw"]

        docs = df["text_raw"].tolist()
        all_docs.extend(docs)

        meta = df[["id", "title", "published", "authors", "link"]].copy()
        meta["year"] = meta["published"].dt.year
        meta["month"] = meta["published"].dt.month
        meta["year_month"] = meta["published"].dt.to_period("M").astype(str)
        meta["year_week"] = meta["published"].dt.to_period("W").astype(str)
        all_meta.append(meta)

        # Save per year if needed
        outname = file.name.replace(".parquet", "")
        df.to_parquet(processed_dir / f"{outname}.parquet", index=False)


    # Save global corpus and metadata to the corpus directory
    with open(corpus_dir / "corpus.txt", "w", encoding="utf-8") as f:
        for line in tqdm(all_docs, desc="Saving corpus"):
            f.write(line.strip() + "\n")

    corpus_df = pd.DataFrame({"text_raw": all_docs})
    corpus_df["id"] = meta["id"]
    corpus_df.to_parquet(corpus_dir / "corpus.parquet", index=False)

    full_meta = pd.concat(all_meta, ignore_index=True)
    full_meta.to_parquet(corpus_dir / "metadata.parquet", index=False)

    job.update_status("preprocessed")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ").replace("\r", " ")
    return text.strip()
