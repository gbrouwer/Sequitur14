# === FILE: tfidf.py ===
# === PURPOSE: Compute TF-IDF keywords using rolling time windows and full dataset ===

import pandas as pd
from datetime import timedelta
from sequitur14.managers import JobManager
from sequitur14.analyzers import TfidfAnalyzer
from utils import load_corpus, save_keywords, load_metadata
from sequitur14.exporters import ResultsExporter
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_MAX_FEATURES = 10000
DEFAULT_N_KEYWORDS = 200

def run_tfidf_timesliced(config):
    """
    Run TF-IDF analysis over time-sliced windows and full dataset.

    Parameters:
        config (dict): Configuration object with keys including:
            - job_name
            - data_name
            - tfidf_max_features
            - n_keywords
            - rolling_window_days
            - rolling_step_days
            - min_docs_per_window
            - stopword_source
    """
    job = JobManager(config, mode="tfidf")
    if not job.check_pipeline_ready():
        raise RuntimeError("Pipeline is not ready. Make sure scraping, staging, and preprocessing are complete.")

    out_dir = job.get_results_path("tfidf")
    job.print_header()

    # Load corpus and metadata
    corpus = load_corpus(config["job_name"])
    metadata = load_metadata(config["job_name"])
    metadata["published"] = pd.to_datetime(metadata["published"], errors="coerce")

    # Load stopwords
    stopwords = set()
    stopword_path = config.get("stopword_source")
    if stopword_path:
        with open(stopword_path, "r", encoding="utf-8") as f:
            stopwords = set(w.strip() for w in f if w.strip())
    print(stopwords)
    max_features = config.get("tfidf_max_features", DEFAULT_MAX_FEATURES)
    n_keywords = config.get("n_keywords", DEFAULT_N_KEYWORDS)

    # === Step 0: Fit global vectorizer ===
    print("\t[Step 0] Fitting global TF-IDF vectorizer")
    vectorizer = TfidfVectorizer(stop_words=None, max_features=max_features)
    vectorizer.fit(corpus)

    # === Step 1: Rolling window TF-IDF ===
    window_size = config.get("rolling_window_days", 30)
    step_size = config.get("rolling_step_days", 7)
    min_docs = config.get("min_docs_per_window", 10)

    start_date = metadata["published"].min()
    end_date = metadata["published"].max()

    time_windows = []
    current_start = start_date
    while current_start + timedelta(days=window_size) <= end_date:
        current_end = current_start + timedelta(days=window_size)
        time_windows.append((current_start, current_end))
        current_start += timedelta(days=step_size)

    print("\t[Step 1] Rolling TF-IDF Analysis:")
    for start, end in tqdm(time_windows, desc="Analyzing windows"):
        mask = (metadata["published"] >= start) & (metadata["published"] < end)
        idxs = metadata[mask].index
        timestamp_str = start.strftime("%Y-%m-%d")

        if len(idxs) < min_docs:
            continue

        docs = [corpus[i] for i in idxs]
        X = vectorizer.transform(docs)
        scores = X.toarray().sum(axis=0)

        keywords = pd.DataFrame({
            "keyword": vectorizer.get_feature_names_out(),
            "score": scores
        }).sort_values(by="score", ascending=False).head(n_keywords)

        if stopwords:
            keywords = keywords[~keywords["keyword"].isin(stopwords)]

        save_keywords(keywords, out_dir / f"keywords_{timestamp_str}.csv")

    # === Step 2: Full corpus TF-IDF ===
    print("\t[Step 2] Analyzing full corpus:")
    X_full = vectorizer.transform(corpus)
    scores = X_full.toarray().sum(axis=0)
    keywords = pd.DataFrame({
        "keyword": vectorizer.get_feature_names_out(),
        "score": scores
    }).sort_values(by="score", ascending=False).head(n_keywords)

    if stopwords:
        keywords = keywords[~keywords["keyword"].isin(stopwords)]

    save_keywords(keywords, out_dir / "keywords_full_corpus.csv")

    print("\tTop 50 keywords in full corpus:")
    print(keywords.head(50).to_string(index=False))

    job.snapshot_to_results()
    print("\tTF-IDF rolling window and full corpus analysis complete.")

"""
run_tfidf_timesliced(): Perform rolling and full corpus TF-IDF keyword extraction.

TF-IDF (Term Frequency–Inverse Document Frequency) is calculated as:

    tfidf(t, d, D) = tf(t, d) * idf(t, D)

Where:
    tf(t, d)   = frequency of term t in document d
    idf(t, D) = log(N / (1 + df(t))) where:
        - N = total number of documents
        - df(t) = number of documents containing term t

Here, we use scikit-learn’s TfidfVectorizer with a fixed vocabulary across time.
For each rolling window, we transform the documents, sum the TF-IDF scores,
rank the terms, and export the top-N keywords. Full corpus TF-IDF is also computed
for baseline comparison and export.
"""
