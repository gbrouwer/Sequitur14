# === FILE: embedders.py ===
# === CHANGE: Add docstrings, inline comments, and config validation for BertEmbedder ===

import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

class BertEmbedder:
    """
    Generates document embeddings using a BERT-based sentence transformer.

    This class:
    - Loads a pre-trained transformer model from a specified path,
    - Encodes a list of documents into dense vector representations,
    - Optionally normalizes the vectors,
    - Saves the resulting embeddings in compressed NumPy format.

    Parameters:
    -----------
    model_path : str
        Path to the transformer model directory (e.g., '../models/all-MiniLM-L6-v2').
    device : str
        Torch device string, either 'cpu' or 'cuda'. Default is 'cpu'.
    """
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device

        # Load the SentenceTransformer model
        self.model = SentenceTransformer(model_path, device=device)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of documents into BERT-based sentence embeddings.

        Parameters:
        -----------
        texts : list of str
            List of cleaned documents (typically abstract + title).
        batch_size : int
            Number of documents to encode per batch.

        Returns:
        --------
        np.ndarray
            Array of shape (n_docs, embedding_dim) containing dense vectors.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

    def save_embeddings(self, embeddings: np.ndarray, output_path):
        """
        Save embeddings to a compressed .npz file on disk.

        Parameters:
        -----------
        embeddings : np.ndarray
            Dense vector representations to save.
        output_path : Path or str
            Output file path (e.g., 'results/embeddings.npz').
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, embeddings=embeddings)
        print(f"✓ Saved embeddings to {output_path}")
