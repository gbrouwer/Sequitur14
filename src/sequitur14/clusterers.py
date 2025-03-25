# === FILE: clusterers.py ===
# === PURPOSE: BERTopic-based clustering utility ===

import pandas as pd
import numpy as np
from bertopic import BERTopic
from umap import UMAP

class BertClusterer:
    def __init__(self, model_path="all-MiniLM-L6-v2", reduce=True, n_topics="auto"):
        """
        Initialize a BERTopic-based clustering model.

        Parameters:
            model_path (str): HuggingFace model path for embeddings
            reduce (bool): Whether to reduce embeddings with UMAP
            n_topics (int or "auto"): Number of topics to extract
        """
        self.reducer_enabled = reduce
        self.n_topics = n_topics
        self.topic_model = BERTopic(
            embedding_model=None,  # we use precomputed embeddings
            umap_model=None if not reduce else UMAP(n_neighbors=15, n_components=2, metric='cosine'),
            nr_topics=n_topics,
            calculate_probabilities=True,
            verbose=False
        )
        self.df_docs = None

    def fit(self, texts, embeddings):
        """
        Fit BERTopic on the full corpus with precomputed embeddings.

        Parameters:
            texts (List[str]): Documents
            embeddings (np.ndarray): Embeddings of the documents
        """
        topics, probs = self.topic_model.fit_transform(texts, embeddings)
        self.df_docs = pd.DataFrame({
            "text": texts,
            "topic": topics,
            "probability": [float(p.max()) if isinstance(p, np.ndarray) else None for p in probs]
        })

        # Add UMAP 2D positions if enabled
        if self.reducer_enabled:
            try:
                fig = self.topic_model.visualize_documents(texts, embeddings=embeddings)
                x_vals = fig['data'][0]['x']
                y_vals = fig['data'][0]['y']
                self.df_docs[["x", "y"]] = pd.DataFrame({"x": x_vals, "y": y_vals})
            except Exception as e:
                print(f"⚠️ UMAP projection failed on full corpus: {e}")
                self.df_docs["x"], self.df_docs["y"] = None, None

    def transform(self, texts, embeddings):
        """
        Apply the fitted BERTopic model to new documents and project into shared 2D space.

        Parameters:
            texts (List[str]): Documents in the time slice
            embeddings (np.ndarray): Their precomputed embeddings

        Returns:
            pd.DataFrame: Document-topic assignments with UMAP coordinates
        """
        topics, probs = self.topic_model.transform(texts, embeddings)
        df = pd.DataFrame({
            "text": texts,
            "topic": topics,
            "probability": [float(p.max()) if isinstance(p, np.ndarray) else None for p in probs]
        })

        # Optional UMAP projection using full model
        if self.reducer_enabled:
            try:
                fig = self.topic_model.visualize_documents(texts, embeddings=embeddings)
                x_vals = fig['data'][0]['x']
                y_vals = fig['data'][0]['y']
                df[["x", "y"]] = pd.DataFrame({"x": x_vals, "y": y_vals})
            except Exception as e:
                print(f"⚠️ UMAP projection failed on slice: {e}")
                df["x"], df["y"] = None, None

        return df
