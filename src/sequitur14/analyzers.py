# === FILE: analyzers.py ===
# === CHANGE: Combine TfidfAnalyzer and LdaAnalyzer in one script, both fully documented ===

import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class TfidfAnalyzer:
    """
    Analyzes a corpus using TF-IDF (Term Frequency–Inverse Document Frequency).

    Responsibilities:
    - Optionally removes stopwords from the corpus.
    - Computes TF-IDF scores across documents.
    - Outputs keyword scores as a DataFrame (CSV and Parquet export supported).

    Parameters:
    -----------
    corpus : list of str
        A list of documents (preprocessed text).
    max_features : int or None
        Max number of terms to retain in the TF-IDF vocabulary.
    stop_words : int, str ("all"), or None
        Number of top stopwords to remove from the custom stopword file.
        If "all", removes all words in the file.
        If None, no stopwords are removed.
    stopword_path : str or Path
        File path to a newline-separated list of stopwords.
    """
    def __init__(self, corpus=None, max_features=None, stop_words=None, stopword_path=None):
        self.corpus = corpus if corpus else []
        self.max_features = max_features
        self.stop_words = stop_words
        self.stopword_path = Path(stopword_path) if stopword_path else None
        self.keywords = []
        self.scores = []
        self.df = pd.DataFrame()
        self.custom_stopwords = self._load_stopwords()

    def _load_stopwords(self):
        """
        Load stopwords from the file based on stop_words setting.
        - If 'all', load entire file.
        - If an integer, load top N lines.
        - If None, return empty list.
        """
        if not self.stopword_path or not self.stopword_path.exists():
            return []

        with open(self.stopword_path, "r", encoding="utf-8") as f:
            full_list = [line.strip().lower() for line in f if line.strip()]

        if self.stop_words == "all":
            return full_list
        elif isinstance(self.stop_words, int):
            return full_list[:self.stop_words]
        else:
            return []

    def preprocess_corpus(self):
        """
        Remove custom stopwords from each document in the corpus.
        """
        if not self.custom_stopwords:
            return self.corpus

        processed = []
        for doc in self.corpus:
            words = doc.split()
            filtered = [word for word in words if word.lower() not in self.custom_stopwords]
            processed.append(" ".join(filtered))
        return processed

    def fit(self):
        """
        Compute TF-IDF scores for all terms in the processed corpus.
        Stores results in a pandas DataFrame with 'Keyword' and 'TF-IDF Score' columns.
        """
        processed_corpus = self.preprocess_corpus()
        print("→ Corpus sample (post-stopword):", processed_corpus[:3])
        print("→ Total docs:", len(processed_corpus))
        print("→ Empty docs:", sum(1 for doc in processed_corpus if not doc.strip()))

        # Compute TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words=None, max_features=self.max_features)
        X = vectorizer.fit_transform(processed_corpus)

        # Aggregate scores for each keyword
        self.keywords = vectorizer.get_feature_names_out()
        self.scores = X.toarray().sum(axis=0)

        # Store results in DataFrame
        self.df = pd.DataFrame({
            'Keyword': self.keywords,
            'TF-IDF Score': self.scores
        })

    def get_top_keywords(self, n=10):
        """
        Return the top n keywords based on TF-IDF score.
        """
        return self.df.sort_values(by='TF-IDF Score', ascending=False).head(n)

    def save_to_csv(self, path):
        """
        Export keyword scores to CSV.
        """
        self.df.to_csv(path, index=False)

    def save_to_parquet(self, path):
        """
        Export keyword scores to Parquet format.
        """
        self.df.to_parquet(path, index=False)

    def print_results(self):
        """
        Print the full TF-IDF results DataFrame.
        """
        print(self.df)


class LdaAnalyzer:
    """
    Performs Latent Dirichlet Allocation (LDA) topic modeling on a document corpus.

    Responsibilities:
    - Tokenize and vectorize documents with a CountVectorizer.
    - Fit an LDA model to identify latent topics.
    - Export top words per topic, keyword frequency, and per-document topic distribution.

    Parameters:
    -----------
    corpus : list of str
        Preprocessed text documents.
    n_topics : int
        Number of topics to infer from the corpus.
    max_features : int
        Maximum vocabulary size used by the vectorizer.
    stop_words : int, str ("all"), or None
        If set, removes the top N stopwords from a given list.
    stopword_path : str or Path
        File path to the stopword list (newline-separated).
    """
    def __init__(self, corpus=None, n_topics=10, max_features=1000, stop_words=None, stopword_path=None):
        self.corpus = corpus if corpus else []
        self.n_topics = n_topics
        self.max_features = max_features
        self.stop_words = stop_words
        self.stopword_path = Path(stopword_path) if stopword_path else None
        self.custom_stopwords = self._load_stopwords()

        self.topics_df = pd.DataFrame()
        self.keywords_df = pd.DataFrame()
        self.document_topics_df = pd.DataFrame()

    def _load_stopwords(self):
        """
        Load and optionally trim a stopword list.
        """
        if not self.stopword_path or not self.stopword_path.exists():
            return []

        with open(self.stopword_path, "r", encoding="utf-8") as f:
            full_list = [line.strip().lower() for line in f if line.strip()]

        if self.stop_words == "all":
            return full_list
        elif isinstance(self.stop_words, int):
            return full_list[:self.stop_words]
        return []

    def preprocess_corpus(self):
        """
        Remove custom stopwords from the corpus.
        """
        if not self.custom_stopwords:
            return self.corpus

        processed = []
        for doc in self.corpus:
            words = doc.split()
            filtered = [word for word in words if word.lower() not in self.custom_stopwords]
            processed.append(" ".join(filtered))
        return processed

    def fit(self):
        """
        Fit the LDA model on the cleaned corpus and extract:
        - Top keywords per topic
        - Keyword frequency across all documents
        - Topic distribution per document
        """
        processed = self.preprocess_corpus()

        # Convert text into a bag-of-words matrix
        vectorizer = CountVectorizer(max_features=self.max_features)
        X = vectorizer.fit_transform(processed)

        # Fit LDA model
        lda = LatentDirichletAllocation(n_components=self.n_topics, random_state=42)
        doc_topic_matrix = lda.fit_transform(X)

        words = vectorizer.get_feature_names_out()

        # Get top keywords for each topic
        topic_rows = []
        for idx, topic in enumerate(lda.components_):
            top_words = [words[i] for i in topic.argsort()[:-11:-1]]
            topic_rows.append({
                "Topic": f"Topic {idx}",
                "Top Keywords": ", ".join(top_words)
            })
        self.topics_df = pd.DataFrame(topic_rows)

        # Compute keyword frequency across all documents
        word_counts = X.toarray().sum(axis=0)
        self.keywords_df = pd.DataFrame({
            "Keyword": words,
            "Frequency": word_counts
        }).sort_values(by="Frequency", ascending=False)

        # Store topic distribution for each document
        topic_cols = [f"Topic {i}" for i in range(self.n_topics)]
        self.document_topics_df = pd.DataFrame(doc_topic_matrix, columns=topic_cols)

    def save_document_topics(self, path):
        """Save per-document topic distribution to CSV."""
        self.document_topics_df.to_csv(path, index=False)

    def save_document_topics_parquet(self, path):
        """Save per-document topic distribution to Parquet."""
        self.document_topics_df.to_parquet(path, index=False)

    def save_topics(self, path):
        """Save top words per topic to CSV."""
        self.topics_df.to_csv(path, index=False)

    def save_keywords(self, path):
        """Save keyword frequency counts to CSV."""
        self.keywords_df.to_csv(path, index=False)

    def save_topics_parquet(self, path):
        """Save top words per topic to Parquet."""
        self.topics_df.to_parquet(path, index=False)

    def save_keywords_parquet(self, path):
        """Save keyword frequency counts to Parquet."""
        self.keywords_df.to_parquet(path, index=False)

    def print_results(self):
        """Print a summary of topic keywords and top global keywords."""
        print("=== LDA Topics ===")
        print(self.topics_df)
        print("\n=== Keyword Frequencies ===")
        print(self.keywords_df.head(10))
