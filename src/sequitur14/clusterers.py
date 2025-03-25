# === FILE: clusterers.py ===
# === PURPOSE: BERTopic-based clustering utility ===

import pandas as pd
import numpy as np
from bertopic import BERTopic

class BertClusterer:
    def __init__(self, model_path, reduce=True, n_topics="auto"):
        self.model_path = model_path
        self.reducer_enabled = reduce
        self.n_topics = n_topics
        nr_topics = None if self.n_topics == "auto" else self.n_topics
        self.topic_model = BERTopic(
            top_n_words=10,
            calculate_probabilities=True,
            verbose=False,
            nr_topics=nr_topics
        )
        self.df_topics = None
        self.df_docs = None

    def fit(self, texts: list[str], embeddings: np.ndarray):
        """
        Fit BERTopic to the provided texts and embeddings.
        Returns:
            df_topics: pd.DataFrame
            df_docs: pd.DataFrame
        """
        topics, probs = self.topic_model.fit_transform(texts, embeddings)

        # Store document-topic assignments and their probabilities
        self.df_docs = pd.DataFrame({
            "text": texts,
            "topic": topics,
            "probability": [float(p.max()) if isinstance(p, np.ndarray) else None for p in probs]
        })

        # Store topic-level summary (including top keywords)
        self.df_topics = self.topic_model.get_topic_info()

        # Optionally compute UMAP layout for visualization
        if self.reducer_enabled:
            try:
                fig = self.topic_model.visualize_documents(texts, embeddings=embeddings)
                x_vals = fig['data'][0]['x']
                y_vals = fig['data'][0]['y']
                self.df_docs[["x", "y"]] = pd.DataFrame({"x": x_vals, "y": y_vals})
            except Exception as e:
                print(f"⚠️ UMAP visualization skipped: {e}")

        return self.df_topics, self.df_docs
