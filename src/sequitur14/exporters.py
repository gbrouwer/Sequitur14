# === FILE: exporters.py ===
# === CHANGE: Add full docstrings, inline comments, and explanations to ResultsExporter ===

import shutil
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict


class ResultsExporter:
    """
    Handles the export of analysis results into a standardized format for visualization.

    Features:
    - Normalizes and relabels output files (e.g., keywords, topics, embeddings).
    - Generates chart configuration JSONs for automated visualization tools.
    - Supports formats from TF-IDF, LDA, and BERTopic pipelines.
    - Also builds radial tree visualizations for topic breakdowns.

    Parameters:
    -----------
    results_dir : Path
        Path to the folder where analysis results were written.
    experiment_name : str
        Name of the experiment (used for folder naming).
    job_hash : str
        Short hash ID for versioning or config fingerprinting.
    data_name : str
        Name of the dataset (used to construct output folder structure).
    """
    def __init__(self, results_dir: Path, experiment_name: str, job_hash: str, data_name: str):
        self.results_dir = Path(results_dir)
        self.experiment_name = experiment_name
        self.job_hash = job_hash
        self.data_name = data_name

        # Construct the output directory for visualization artifacts
        self.viz_dir = Path("../viz/results") / data_name / experiment_name / job_hash
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def export_all(self):
        """
        Export all CSV results and matching chart config JSONs.
        - Renames and standardizes files from the results directory.
        - Generates specific configs based on filename content.
        - Optionally copies over the experiment config.yaml.
        """
        print(f"→ Exporting results to: {self.viz_dir}")
        exported = 0

        for item in self.results_dir.glob("**/*.csv"):
            if item.is_file():
                df = pd.read_csv(item)
                df.columns = [col.strip().lower() for col in df.columns]  # Normalize headers
                config = {}

                # TF-IDF output (keywords + scores)
                if item.stem == "keywords" and "keyword" in df.columns and "tf-idf score" in df.columns:
                    df = df.rename(columns={"keyword": "key", "tf-idf score": "value"})
                    config = {
                        "x_label": "Keyword",
                        "y_label": "TF-IDF Score",
                        "chart_type": "bar",
                        "color": "#d62728",
                        "caption": "TF-IDF keyword distribution"
                    }

                # LDA output (keywords + frequency)
                elif item.stem == "keywords" and "keyword" in df.columns and "frequency" in df.columns:
                    df = df.rename(columns={"keyword": "key", "frequency": "value"})
                    df["value"] = df["value"] / df["value"].max()  # Normalize for plotting
                    config = {
                        "x_label": "Keyword",
                        "y_label": "Normalized Frequency",
                        "chart_type": "bar",
                        "color": "#2ca02c",
                        "caption": "Top keywords extracted from LDA with frequency normalized"
                    }

                # LDA topic-term relevance (tree data)
                elif item.stem == "topics" and {"term", "relevance", "topic_id"}.issubset(df.columns):
                    df = df.rename(columns={"term": "key", "relevance": "value"})
                    config = {
                        "x_label": "Term",
                        "y_label": "Topic Relevance",
                        "chart_type": "bar",
                        "group_by": "topic_id",
                        "color": "#2ca02c",
                        "caption": "Top terms per topic discovered by LDA"
                    }

                    # Generate radial topic tree JSON for hierarchical view
                    tree_data = self.build_topic_tree(df)
                    with open(self.viz_dir / "topic_tree.json", "w") as f:
                        json.dump(tree_data, f, indent=2)
                    with open(self.viz_dir / "topic_tree_config.json", "w") as f:
                        json.dump({
                            "chart_type": "radial",
                            "caption": "Hierarchical topic breakdown detected via LDA",
                            "color": "#2ca02c"
                        }, f, indent=2)
                    print("🌳 Exported topic_tree.json and config")

                # Document-topic distributions (from LDA or BERTopic)
                elif item.stem == "document_topics" and {"document", "topic"}.issubset(df.columns):
                    df = df.rename(columns={"document": "key", "topic": "value"})
                    config = {
                        "x_label": "Document",
                        "y_label": "Topic Assignment",
                        "chart_type": "bar",
                        "color": "#1f77b4",
                        "caption": "Topic assignments per document (BERT or LDA)"
                    }

                # 2D layout for embeddings (UMAP or t-SNE)
                elif item.stem in ["umap_2d", "tsne_2d"] and {"x", "y"}.issubset(df.columns):
                    config = {
                        "x_label": "x",
                        "y_label": "y",
                        "chart_type": "scatter",
                        "color": "#1f77b4",
                        "caption": f"{item.stem.upper()} layout of document embeddings"
                    }

                # Export standardized CSV file
                csv_path = self.viz_dir / item.name
                df.to_csv(csv_path, index=False)
                exported += 1

                # Export matching JSON config (if relevant)
                if config:
                    config_name = f"{item.stem}_config.json"
                    with open(self.viz_dir / config_name, "w") as f:
                        json.dump(config, f, indent=2)
                    print(f"📝 Wrote config: {config_name}")

        # Copy config.yaml (if found) for experiment reproducibility
        config_path = self.results_dir.parent / "config.yaml"
        if config_path.exists():
            shutil.copy(config_path, self.viz_dir / "config.yaml")

        print(f"✓ Exported {exported} standardized CSV files with configs where applicable.")

    def build_topic_tree(self, df: pd.DataFrame) -> dict:
        """
        Construct a nested JSON object for radial topic tree visualization.

        Parameters:
        -----------
        df : pd.DataFrame
            Must contain 'topic_id', 'key', and 'value' columns (renamed from 'term' and 'relevance').

        Returns:
        --------
        dict
            Hierarchical topic → term structure suitable for D3-style radial layout.
        """
        tree = {"name": "Detected Topics", "children": []}
        grouped = defaultdict(list)

        for _, row in df.iterrows():
            grouped[int(row["topic_id"])].append({
                "name": row["key"],
                "value": float(row["value"])
            })

        for topic_id, keywords in grouped.items():
            tree["children"].append({
                "name": f"Topic {topic_id}",
                "children": keywords
            })

        return tree
