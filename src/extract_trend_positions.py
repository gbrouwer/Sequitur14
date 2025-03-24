# === FILE: extract_topic_positions.py ===
# === PURPOSE: Extract UMAP/2D positions of topics over time from BERTopic clusterer ===

import pandas as pd
from pathlib import Path

# === CONFIG ===
RESULTS_DIR = Path("../results/bert/arxiv-ai-mvp-monthly")  # Update if needed

data = []

# === Iterate through monthly topic files ===
for path in sorted(RESULTS_DIR.glob("topics_*.csv")):
    month = path.stem.split("_")[-1]
    df = pd.read_csv(path)

    # Some BERTopic versions export UMAP coords as 'x' and 'y'; fallback if needed
    if {'x', 'y'}.issubset(df.columns):
        for _, row in df.iterrows():
            data.append({
                "month": month,
                "topic_id": row['Topic'],
                "keyword": row['Keyword'],
                "score": row['Score'],
                "x": row['x'],
                "y": row['y']
            })
    else:
        print(f"⚠️ No UMAP coordinates found in {path.name}, skipping...")

# === Export Combined File ===
if data:
    out_df = pd.DataFrame(data)
    out_df.to_csv("../results/topic_positions_2D.csv", index=False)
    print("✅ Topic positions saved to topic_positions_2D.csv")
else:
    print("❌ No topic UMAP positions were found across time slices.")
