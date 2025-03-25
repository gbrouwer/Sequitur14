# === FILE: lda.py ===
# === PURPOSE: Compute LDA topics using rolling time windows and full dataset ===

import pandas as pd
from datetime import timedelta
from sequitur14.managers import JobManager
from sequitur14.analyzers import GensimLdaAnalyzer
from utils import load_corpus, load_metadata, save_topics, save_document_topics
from tqdm import tqdm
from gensim.models import CoherenceModel

DEFAULT_MAX_FEATURES = 10000
DEFAULT_N_KEYWORDS = 200
DEFAULT_N_TOPICS = 25


def run_lda_timesliced(config):
    job = JobManager(config=config, mode="lda")
    if not job.check_pipeline_ready():
        raise RuntimeError(
            "Pipeline is not ready. Make sure scraping, staging, and preprocessing are complete."
        )

    job.print_header()
    out_dir = job.get_results_path("lda")

    corpus = load_corpus(config["job_name"])
    metadata = load_metadata(config["job_name"])
    metadata["published"] = pd.to_datetime(metadata["published"], errors="coerce")

    window_size = config.get("rolling_window_days", 30)
    step_size = config.get("rolling_step_days", 7)
    min_docs = config.get("min_docs_per_window", 10)
    max_features = config.get("tfidf_max_features", DEFAULT_MAX_FEATURES)
    n_topics = config.get("n_topics", DEFAULT_N_TOPICS)

    print("\n[Step 1] Fitting full corpus LDA model (vocabulary anchor)")
    full_analyzer = GensimLdaAnalyzer(
        corpus=corpus,
        n_topics=n_topics,
        max_features=max_features,
        stop_words=config.get("remove_top_n_stopwords"),
        stopword_path=config.get("stopword_source"),
    )
    full_analyzer.fit()

    save_topics(full_analyzer.topics_df, out_dir / "topics_full_corpus.csv")
    save_document_topics(
        full_analyzer.document_topics_df, out_dir / "doc_topics_full_corpus.csv"
    )

    print("\n🔝 Top words in each topic (full corpus):")
    for topic_id in range(n_topics):
        top_words = full_analyzer.topics_df.query(f"topic == {topic_id}").nlargest(
            10, "score"
        )
        keywords = ", ".join(top_words["keyword"].tolist())
        print(f"Topic {topic_id:2d}: {keywords}")

    print("\n[Step 2] Rolling LDA Analysis (inference only):")
    current_start = metadata["published"].min()
    end_date = metadata["published"].max()

    while current_start + timedelta(days=window_size) <= end_date:
        current_end = current_start + timedelta(days=window_size)
        mask = (metadata["published"] >= current_start) & (
            metadata["published"] < current_end
        )
        idxs = metadata[mask].index

        timestamp_str = current_start.strftime("%Y-%m-%d")

        if len(idxs) < min_docs:
            current_start += timedelta(days=step_size)
            continue

        docs = [corpus[i] for i in idxs]

        # Inference using full model
        tokenized = GensimLdaAnalyzer(
            docs, dictionary=full_analyzer.dictionary
        ).preprocess()
        bow = [full_analyzer.dictionary.doc2bow(toks) for toks in tokenized]
        topic_distributions = [
            [
                prob
                for _, prob in full_analyzer.lda_model.get_document_topics(
                    b, minimum_probability=0.0
                )
            ]
            for b in bow
        ]
        doc_topics_df = pd.DataFrame(topic_distributions)
        doc_topics_df.columns = [f"topic_{i}" for i in range(n_topics)]

        save_document_topics(doc_topics_df, out_dir / f"doc_topics_{timestamp_str}.csv")

        current_start += timedelta(days=step_size)

    job.snapshot_to_results()
    print("\n✅ LDA rolling window and full corpus analysis complete.")


# === Optional helper to explore optimal number of topics ===
def evaluate_topic_coherence(corpus, config, topic_range=(5, 50, 5)):
    print("Evaluating topic coherence across topic counts:")
    results = []
    stop_words = config.get("remove_top_n_stopwords")
    stopword_path = config.get("stopword_source")
    max_features = config.get("tfidf_max_features", DEFAULT_MAX_FEATURES)

    for n in range(topic_range[0], topic_range[1] + 1, topic_range[2]):
        analyzer = GensimLdaAnalyzer(
            corpus=corpus,
            n_topics=n,
            max_features=max_features,
            stop_words=stop_words,
            stopword_path=stopword_path,
        )
        analyzer.fit()

        cm = CoherenceModel(
            model=analyzer.lda_model,
            texts=analyzer.preprocess(),
            dictionary=analyzer.dictionary,
            coherence="c_v",
        )
        score = cm.get_coherence()
        print(f"n_topics = {n:2d} → coherence = {score:.4f}")
        results.append((n, score))

    return results
