# === FILE: main.py ===
# === PURPOSE: Run full end-to-end pipeline with monthly scraping for MVP ===

from scraper import scrape_arxiv
from stager import stage_arxiv_data
from preprocess import preprocess_from_staging
from tfidf import run_tfidf_timesliced
from lda import run_lda_timesliced
from bert import run_bert_timesliced
from sequitur14.managers import JobManager

config = {
    "job_name": "arxiv-ai-mvp-monthly",
    "data_name": "arxiv-cs-ai-2005-2023-monthly",
    "data_source": "arxiv",
    "category": "cs.AI",
    "start_year": 2000,
    "end_year": 2025,
    "max_results_per_month": 250,
    "expand_contractions": True,
    "sampling_freq": "monthly",
    "embedding_model": "../models/all-MiniLM-L6-v2"
}

# ---- Step 0: Scrape ArXiv Data ----
scrape_arxiv(config)

# ---- Step 1: Stage Scraped Data ----
stage_arxiv_data(config)

# ---- Step 2: Preprocessing Text ----
print("\n=== Step 2: Preprocessing Text ===")
preprocess_from_staging(config)

# ---- Step 2.5: Save Preprocessing Config (needed for downstream JobManager) ----
print("\n=== Step 2.5: Saving Preprocessing Config ===")
JobManager(config, mode="preprocess")

# ---- Step 3: Time-Sliced TF-IDF Analysis ----
print("\n=== Step 3: Time-Sliced TF-IDF Analysis ===")
run_tfidf_timesliced(config)

# ---- Step 4: LDA Topic Modeling ----
print("\n=== Step 4: LDA Topic Modeling ===")
run_lda_timesliced(config)

# ---- Step 5: Time-Sliced BERTopic Clustering ----
print("\n=== Step 5: Time-Sliced BERTopic Clustering ===")
run_bert_timesliced(config)

print("\n✓ MVP pipeline complete. All analyses finished.")
