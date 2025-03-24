# === FILE: clusterers.py ===
# === CHANGE: Add full docstrings, inline comments, type hints to BertClusterer ===

import numpy as np
import pandas as pd
from pathlib import Path
from bertopic import BERTopic
from sklearn.manifold import TSNE
import joblib

class BertClusterer:
    """
    Clusters documents using BERTopic, with optional UMAP and t-SNE visualizations.

    This class:
    - Applies BERTopic to BERT-based embeddings and original texts,
    - Stores both topic-level and document-level outputs,
    - Saves interactive visualizations (UMAP, t-SNE),
    - Optionally maps between UMAP and t-SNE spaces using a VisualMapper.

    Parameters:
    -----------
    model_path : str
        Path to a sentence transformer model (used by BERTopic internally).
    reduce : bool
        Whether to generate UMAP-based 2D embeddings for visualization.
    n_topics : int or "auto"
        Number of topics to extract. "auto" lets BERTopic determine optimal count.
    """
    def __init__(self, model_path: str, reduce: bool = True, n_topics="auto"):
        self.topic_model = BERTopic(
            embedding_model=model_path,
            calculate_probabilities=True,
            nr_topics=n_topics,
            low_memory=True,
            verbose=False
        )
        self.df_topics = pd.DataFrame()
        self.df_docs = pd.DataFrame()
        self.reducer_enabled = reduce

    def fit(self, texts: list[str], embeddings: np.ndarray):
        """
        Fit BERTopic to the provided texts and embeddings.

        Parameters:
        -----------
        texts : list of str
            Cleaned documents (e.g., title + abstract).
        embeddings : np.ndarray
            Precomputed sentence embeddings from a BERT model.
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
            fig = self.topic_model.visualize_documents(texts, embeddings=embeddings)
            x_vals = fig['data'][0]['x']
            y_vals = fig['data'][0]['y']
            self.df_docs[["x", "y"]] = pd.DataFrame({"x": x_vals, "y": y_vals})

    def save_results(self, results_dir: Path, embeddings: np.ndarray = None):
        """
        Save topics, document assignments, and visualizations to disk.

        Parameters:
        -----------
        results_dir : Path
            Directory where results should be saved.
        embeddings : np.ndarray or None
            If provided, a 2D t-SNE projection will also be saved.
        """
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save topic-level and document-level results
        self.df_topics.to_csv(results_dir / "topics.csv", index=False)
        self.df_docs.to_csv(results_dir / "document_topics.csv", index=False)

        # Save UMAP 2D layout if available
        if "x" in self.df_docs.columns and "y" in self.df_docs.columns:
            umap_2d = self.df_docs[["x", "y"]]
            umap_2d.to_csv(results_dir / "umap_2d.csv", index=False)

        # Optionally compute and save t-SNE layout
        if embeddings is not None:
            tsne_model = TSNE(n_components=2, perplexity=30, random_state=42)
            tsne_coords = tsne_model.fit_transform(embeddings)
            tsne_df = pd.DataFrame(tsne_coords, columns=["x", "y"])
            tsne_df.to_csv(results_dir / "tsne_2d.csv", index=False)

            # Optional: map UMAP to t-SNE coordinates using VisualMapper
            if "x" in self.df_docs.columns and "y" in self.df_docs.columns:
                umap_2d = self.df_docs[["x", "y"]].values
                tsne_2d = tsne_df.values
                from pulse020.mappers import VisualMapper
                mapper = VisualMapper()
                mapper.fit(umap_coords=umap_2d, tsne_coords=tsne_2d)
                joblib.dump(mapper, results_dir / "umap_to_tsne_mapper.joblib")

        print(f"✓ Saved clustering results to: {results_dir}")
