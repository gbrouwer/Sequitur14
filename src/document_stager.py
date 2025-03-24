# === FILE: stagers.py ===
# === PURPOSE: Stage raw scraped arXiv data into structured yearly Parquet files ===

import pandas as pd
import hashlib
from pathlib import Path
from tqdm import tqdm

def stage_arxiv_data(config):
    """
    Converts raw CSVs from scraping into structured annual Parquet files.
    Output is stored in ../data/<data_name>/staging
    """
    raw_dir = Path("../data") / config["data_name"] / "raw"
    staging_dir = Path("../data") / config["data_name"] / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    grouped = {}
    print(f"📂 Loading raw files from: {raw_dir}")

    for file in tqdm(sorted(raw_dir.glob("raw_*.csv"))):
        df = pd.read_csv(file)

        # Standardize column names
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

        # Add hash column for deduplication
        df['hash'] = df.apply(_hash_row, axis=1)

        # Parse publication dates
        df['published'] = pd.to_datetime(df['published'], errors='coerce')
        df['year'] = df['published'].dt.year
        df['month'] = df['published'].dt.month
        df['year_month'] = df['published'].dt.strftime('%Y-%m')

        # Group by year
        for year, df_group in df.groupby('year'):
            if pd.isna(year):
                continue
            if year not in grouped:
                grouped[year] = []
            grouped[year].append(df_group)

    # Save each year's data as a single Parquet file
    for year, parts in grouped.items():
        full_df = pd.concat(parts, ignore_index=True)
        outpath = staging_dir / f"arxiv_{year}.parquet"
        full_df.to_parquet(outpath, index=False)
        print(f"✓ Staged {len(full_df)} records for year {year}")

def _hash_row(row):
    raw_string = '|'.join(str(row[col]) for col in row.index)
    return hashlib.sha256(raw_string.encode('utf-8')).hexdigest()