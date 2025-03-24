# === FILE: preprocessors.py ===
# === PURPOSE: Preprocess and clean document text and metadata ===

import re
from typing import List
import string
import unicodedata
import contractions
import pandas as pd
from tqdm import tqdm

from pathlib import Path
from utils import save_corpus
from utils import save_metadata

def clean_text(text: str, expand_contractions: bool = True) -> str:
    if expand_contractions:
        text = contractions.fix(text)

    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    return text.strip()

def preprocess_from_staging(config: dict) -> None:
    """
    Load documents from a parquet file in the staging directory, preprocess them,
    and save the output to disk.

    Parameters:
    -----------
    config : dict
        Must include 'staging_dir' and 'processed' directory paths.
    """
    cleaned = []
    metadata = []
    staging_path = Path("../data") / config["data_name"] / "staging"
    processed_dir = Path("../data") / config["data_name"] / "processed"
    for file in tqdm(sorted(staging_path.glob("arxiv_*.parquet"))):
        df = pd.read_parquet(file)
        for _, doc in df.iterrows():
            cleaned_text = clean_text(doc["title"] + " " + doc["summary"], config.get("expand_contractions", True))
            cleaned.append(cleaned_text)
            metadata.append({
                "id": doc.get("id"),
                "title": doc.get("title"),
                "published": doc.get("published")
            })

    processed_dir.mkdir(parents=True, exist_ok=True)
    save_corpus(cleaned, processed_dir / "corpus.txt")
    save_metadata(metadata, processed_dir / "metadata.parquet")