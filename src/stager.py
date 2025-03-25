# === FILE: stager.py ===
# === PURPOSE: Stage raw scraped arXiv data into structured yearly Parquet files ===

import pandas as pd
import hashlib
from pathlib import Path
from tqdm import tqdm
from sequitur14.managers import JobManager


def stage_arxiv_data(config):
    job = JobManager(config, mode="stage")
    raw_dir = job.get_data_path("raw")
    staging_dir = job.get_data_path("staging")

    grouped = {}
    print(f"📂 Loading raw files from: {raw_dir}")

    raw_files = sorted(raw_dir.glob("raw_*.csv"))

    for file in tqdm(raw_files, desc="Staging raw files"):
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

    for year, parts in tqdm(grouped.items(), desc="Saving yearly Parquet files"):
        full_df = pd.concat(parts, ignore_index=True)
        outpath = staging_dir / f"arxiv_{year}.parquet"
        full_df.to_parquet(outpath, index=False)

    job.update_status("staged")


def _hash_row(row):
    raw_string = '|'.join(str(row[col]) for col in row.index)
    return hashlib.sha256(raw_string.encode('utf-8')).hexdigest()