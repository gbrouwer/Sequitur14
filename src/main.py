# === FILE: main.py ===

from scraper import scrape_arxiv
from stager import stage_arxiv_data
from preprocess import preprocess_from_staging
from tfidf import run_tfidf_timesliced
from lda import run_lda_timesliced
from bert import run_bert_timesliced
from sequitur14.managers import JobManager
# from extract_trend_topics import extract_trend_topics
# from extract_topic_positions import extract_topic_positions

config = {
    "job_name": "arxiv-ai-mvp-weekly",
    "data_name": "arxiv-cs-ai-2010-2025-weekly",
    "data_source": "arxiv",
    "category": "cs.AI",
    "start_year": 2010,
    "end_year": 2025,
    "max_results_per_interval": 300,
    "expand_contractions": True,
    "sampling_freq": "weekly",
    "tfidf_max_features": None,
    "n_keywords": 500,
    "no_topics": 50,
    "remove_top_n_stopwords": "all",
    "stopword_source": "../meta/stopwords/stopwords_general_and_scientific_english_plus.txt",
    "embedding_model": "../models/all-MiniLM-L6-v2",
    "device": "cuda",
    "min_docs_per_slice": 15,
    "rolling_window_days": 30,
    "rolling_step_days": 7,
    "min_docs_per_window": 10,
}

# Initialize and save config
print("[Step 0] Set Config")
job = JobManager(config, mode="init", force=True)
job.update_status("config_initialized")

# # Step 0: Scrape
print("[Step 1] Scraping data")
# scrape_arxiv(config)
# job.update_status("scraped")

# Step 1: Stage
print("[Step 2] Staging data")
# stage_arxiv_data(config)
# job.update_status("staged")

# # Step 2: Preprocess
print("[Step 3] Initial Preprocess of data")
# preprocess_from_staging(config)
# job.update_status("preprocessed")
# job.update_status("corpus")

# Step 3: TF-IDF
print("[Step 4] TFiDF Analysis")
# run_tfidf_timesliced(config)

# # Q Step 4: LDA
print("[Step 5] LDA Analysis")
# run_lda_timesliced(config)

# # Step 5: BERTopic
print("[Step 6] BERTopic Analyis")
run_bert_timesliced(config)

# # Step 6: Extract LLM Trend Data
# print("\n=== Step 6: Extracting LLM Trending Topics ===")
# extract_trend_topics()

# # Step 7: Extract UMAP 2D Topic Positions
# print("\n=== Step 7: Extracting Topic Positions in 2D ===")
# extract_topic_positions()

# === Step 8: Export Results to Dashboard ===
# from sequitur14.exporters import ResultsExporter
#
# exporter = ResultsExporter(config, config["data_name"])
# exporter.run()
#
# print("\n🎉 All steps complete. Results saved to /results")
