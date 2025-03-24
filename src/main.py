# === FILE: main.py ===

from scraper import scrape_arxiv
from stager import stage_arxiv_data
from preprocess_documents import preprocess_from_staging
from tfidf import run_tfidf_timesliced
from lda import run_lda_timesliced
from bert_ import run_bert_timesliced
from sequitur14.managers import JobManager
from extract_trend_topics import extract_trend_topics
from extract_trend_positions import extract_topic_positions

config = {
    "job_name": "arxiv-ai-mvp-monthly",
    "data_name": "arxiv-cs-ai-2005-2023-monthly",
    "data_source": "arxiv",
    "category": "cs.AI",
    "start_year": 2020,
    "end_year": 2021,
    "max_results_per_month": 250,
    "expand_contractions": True,
    "sampling_freq": "monthly",
    "tfidf_max_features": 500,
    "remove_top_n_stopwords": "all",
    "stopword_source": "../meta/stopwords_general_and_scientific_english.txt",
    "embedding_model": "../models/all-MiniLM-L6-v2",
    "device": "cpu"
}

# Step 0: Scrape
scrape_arxiv(config)

# Step 1: Stage
stage_arxiv_data(config)

# Step 2: Preprocess
preprocess_from_staging(config)

# Step 2.5: Save config
JobManager(config, mode="preprocess")

# Step 3: TF-IDF
run_tfidf_timesliced(config)

# Step 4: LDA
run_lda_timesliced(config)

# Step 5: BERTopic
run_bert_timesliced(config)

# Step 6: Extract LLM Trend Data
print("\n=== Step 6: Extracting LLM Trend Topics ===")
extract_trend_topics()

# Step 7: Extract UMAP 2D Topic Positions
print("\n=== Step 7: Extracting Topic Positions in 2D ===")
extract_topic_positions()

print("\n🎉 All steps complete. Results saved to /results")
