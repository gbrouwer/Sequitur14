# Sequitur14: AI Research Trend Detection Pipeline

Sequitur14 is a modular pipeline designed to detect emerging trends in AI research using arXiv data. It scrapes, preprocesses, and analyzes time-sliced scientific papers using TF-IDF, LDA, and BERTopic, with follow-up trend and topic visualization tools.

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/sequitur14.git
cd sequitur14
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
python main.py
```

The pipeline will:
1. Scrape and structure arXiv data
2. Preprocess and clean the text
3. Run TF-IDF, LDA, and BERTopic over time
4. Extract topic trends and coordinates
5. Save everything into `/data/` and `/results/` folders

---

## 🗂️ Project Structure

```
Sequitur14/
├── src/
│   ├── main.py
│   ├── scraper.py
│   ├── stager.py
│   ├── preprocess.py
│   ├── tfidf.py
│   ├── lda.py
│   ├── bert.py
│   ├── extract_trend_topics.py
│   ├── extract_topic_positions.py
│   └── sequitur14/
│       ├── managers.py
│       ├── analyzers.py
│       ├── embedders.py
│       └── clusterers.py
├── data/
├── results/
├── models/
├── meta/
└── requirements.txt
```

---

## 📄 Script Descriptions

### `main.py`
Runs the full pipeline end-to-end. Inputs a config dict, outputs all intermediate and final results to `data/` and `results/`.

### `scraper.py`
Scrapes arXiv using the API. Outputs monthly raw CSVs into `data/<data_name>/raw/`.

### `stager.py`
Cleans and groups raw data into yearly Parquet files under `staging/`.

### `preprocess.py`
Processes title and abstract into cleaned text corpus + metadata. Saves to `processed/`.

### `tfidf.py`
Performs TF-IDF keyword extraction per time slice. Outputs one CSV per month.

### `lda.py`
Runs LDA topic modeling by month. Outputs topic words and doc-topic distributions.

### `bert.py`
Executes BERTopic clustering per time slice using transformer embeddings.

### `extract_trend_topics.py`
Scans BERTopic output to track LLM-related topic emergence. Saves to `trend_summary_llm_vs_baseline.csv`.

### `extract_topic_positions.py`
Pulls UMAP 2D topic coordinates from BERTopic results. Saves to `topic_positions_2D.csv`.

---

## 🧠 Dependencies

- pandas, numpy, nltk, contractions
- scikit-learn, umap-learn
- BERTopic, sentence-transformers
- tqdm, feedparser

---

## 📝 License

MIT — built for rapid prototyping and research.

---
