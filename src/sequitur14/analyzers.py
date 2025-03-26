import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim import corpora
from gensim.models import LdaModel
from gensim.matutils import Sparse2Corpus
import numpy as np
import re


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

    def __init__(
        self, corpus=None, max_features=None, stop_words=None, stopword_path=None
    ):
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
            return full_list[: self.stop_words]
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
            filtered = [
                word for word in words if word.lower() not in self.custom_stopwords
            ]
            processed.append(" ".join(filtered))
        return processed

    def fit(self):
        """
        Compute TF-IDF scores for all terms in the processed corpus.
        Stores results in a pandas DataFrame with 'Keyword' and 'TF-IDF Score' columns.
        """
        processed_corpus = self.preprocess_corpus()
        # print("→ Corpus sample (post-stopword):", processed_corpus[0])
        # print("→ Total docs:", len(processed_corpus))
        # print("→ Empty docs:", sum(1 for doc in processed_corpus if not doc.strip()))

        # Compute TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words=None, max_features=self.max_features)
        X = vectorizer.fit_transform(processed_corpus)

        # Aggregate scores for each keyword
        self.keywords = vectorizer.get_feature_names_out()
        self.scores = X.toarray().sum(axis=0)

        # Store results in DataFrame
        self.df = pd.DataFrame({"Keyword": self.keywords, "TF-IDF Score": self.scores})

    def get_top_keywords(self, n=10):
        """
        Return the top n keywords based on TF-IDF score.
        """
        return self.df.sort_values(by="TF-IDF Score", ascending=False).head(n)

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


class GensimLdaAnalyzer:
    def __init__(
        self,
        corpus,
        n_topics=10,
        max_features=10000,
        stop_words=None,
        stopword_path=None,
        dictionary=None,
    ):
        self.raw_docs = corpus
        self.n_topics = n_topics
        self.max_features = max_features
        self.stop_words = set(stop_words or [])
        if stopword_path:
            with open(stopword_path, "r") as f:
                self.stop_words.update(w.strip() for w in f.readlines())
        self.dictionary = dictionary  # optional precomputed Gensim dictionary
        self.lda_model = None
        self.topics_df = None
        self.document_topics_df = None

    def preprocess(self):
        tokenized = []
        for doc in self.raw_docs:
            doc = doc.lower()
            doc = re.sub(r"[\n\r\t]", " ", doc)
            doc = re.sub(r"\s+", " ", doc)
            tokens = doc.strip().split(" ")
            tokens = [t for t in tokens if len(t) > 1 and t not in self.stop_words]
            tokenized.append(tokens)
        return tokenized

    def fit(self):
        tokenized_docs = self.preprocess()

        # Build or reuse dictionary
        if self.dictionary is None:
            self.dictionary = corpora.Dictionary(tokenized_docs)
            self.dictionary.filter_extremes(
                no_below=5, no_above=0.5, keep_n=self.max_features
            )

        bow_corpus = [self.dictionary.doc2bow(doc) for doc in tokenized_docs]

        self.lda_model = LdaModel(
            corpus=bow_corpus,
            id2word=self.dictionary,
            num_topics=self.n_topics,
            passes=10,
            random_state=42,
        )

        # Extract topic terms, ensure all topics represented
        self.topics_df = pd.DataFrame(
            [
                {"topic": i, "keyword": word, "score": float(prob)}
                for i in range(self.n_topics)
                for word, prob in self.lda_model.show_topic(i, topn=20)
            ]
        )

        # Extract document-topic distribution (safe against missing topics)
        all_distributions = []
        for bow in bow_corpus:
            dist = dict(
                self.lda_model.get_document_topics(bow, minimum_probability=0.0)
            )
            row = [dist.get(i, 0.0) for i in range(self.n_topics)]
            all_distributions.append(row)

        self.document_topics_df = pd.DataFrame(
            all_distributions, columns=[f"topic_{i}" for i in range(self.n_topics)]
        )
