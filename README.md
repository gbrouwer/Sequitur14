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

The scrapers.py module is responsible for fetching AI-related research articles from the arXiv API across a defined time range and category (e.g., cs.AI). It segments the time range into smaller intervals (monthly, weekly, or daily), and for each slice, it constructs a properly formatted query that includes the time window using submittedDate and the specified category. It then sends the request to the arXiv API, parses the results, and saves each batch of articles to a dedicated CSV file in a raw/ directory. This approach ensures data is incrementally saved for each slice, making the process robust to failures and allowing the pipeline to resume from previously scraped points. This modular and extensible design supports scalable and resumable large-scale data collection from arXiv.

### `stager.py`
Cleans and groups raw data into yearly Parquet files under `staging/`.

The stagers.py module is responsible for transforming raw scraped data from arXiv into a clean, uniform intermediate format that can be used by downstream processing steps. It loads the individual raw CSV files (one per time slice) from the raw/ directory, extracts relevant fields (such as title, abstract, published date, and ID), and filters or deduplicates entries as needed. The processed documents are then bundled together and saved as a single staged.jsonl file, where each line contains one JSON-encoded document. This staging process ensures that all documents are in a consistent structure and format, decoupling the variability of raw input data from the rest of the pipeline and improving overall modularity and robustness.

### `preprocess.py`
Processes title and abstract into cleaned text corpus + metadata. Saves to `processed/`.

The preprocessors.py module is responsible for cleaning and standardizing the text data from arXiv documents in preparation for analysis. It loads the staged data (typically from a .parquet or .jsonl file), processes each document by combining the title and abstract, and applies a series of text normalization steps such as lowercasing, removing punctuation, expanding contractions, removing stopwords, and stemming. This results in a clean, tokenized version of the text that is better suited for NLP techniques like TF-IDF, LDA, or BERT. In addition to producing the cleaned corpus, the module extracts and saves associated metadata—such as the document ID, title, and published date—which is essential for later steps like time slicing and trend detection. This preprocessing step ensures consistency and quality across all documents before they enter the analysis phase.

### `tfidf.py`
Performs TF-IDF keyword extraction per time slice. Outputs one CSV per month.

TF-IDF (Term Frequency–Inverse Document Frequency) is a widely used metric in information retrieval and natural language processing that quantifies how important a word is to a particular document relative to a collection (or corpus) of documents. It helps to identify words that are both frequent in a document and distinctive across the corpus, making it useful for keyword extraction, topic modeling, and text classification.

The formula for TF-IDF of a term t in a document d is:

![TF-IDF full](https://latex.codecogs.com/svg.image?TF\text{-}IDF(t,&space;d)&space;=&space;\frac{f_{t,d}}{\sum_k&space;f_{k,d}}&space;\times&space;\log\left(&space;\frac{N}{1&plus;n_t}&space;\right))

Where:
- TF(t, d)   = term frequency of term t in document d
             = f_{t,d} / sum_k f_{k,d}
             (f_{t,d} is the number of times term t appears in document d,
              sum_k f_{k,d} is the total number of terms in document d)

- IDF(t)     = inverse document frequency of term t
             = log(N / (1 + n_t))
             (N is the total number of documents,
              n_t is the number of documents that contain term t)

- TF-IDF(t, d) = TF(t, d) * IDF(t)

The result is a score that’s high when a word is common in one document but rare across the corpus — a strong signal of that word’s specificity or relevance.

The run_tfidf.py module applies the TF-IDF metric to a corpus of preprocessed arXiv documents in order to identify the most significant keywords across the entire dataset. It uses scikit-learn’s TfidfVectorizer under the hood, and is configurable in terms of the number of features to extract, custom stopword lists, and whether or not to remove the top N most common terms. Once the model is fit on the corpus, it ranks terms for each document or across the entire dataset based on their TF-IDF scores. The resulting keyword lists are saved in structured CSV files, making them suitable for downstream use in trend detection, topic modeling, or visualization. This module is a foundational piece of the pipeline’s keyword analysis layer.

### `lda.py`
Runs LDA topic modeling by month. Outputs topic words and doc-topic distributions.

Latent Dirichlet Allocation (LDA) is a probabilistic topic modeling technique used to discover abstract themes (or "topics") within a collection of documents. It assumes each document is a mixture of multiple topics, and each topic is a probability distribution over words. LDA attempts to uncover these hidden topic structures based on the patterns of word usage across documents, without requiring any prior labeling.

LDA models the following generative process for each document d in a corpus:

1. Choose a distribution over topics:       θ_d ~ Dirichlet(α)
2. For each word w in document d:
   a. Choose a topic z ~ Multinomial(θ_d)
   b. Choose a word  w ~ Multinomial(β_z)

Where:
- θ_d: Topic distribution for document d
- z: Chosen topic for a given word
- w: Word in the document
- β_z: Word distribution for topic z
- α, β: Hyperparameters controlling topic and word sparsity


The run_lda.py module applies LDA topic modeling to a corpus of preprocessed arXiv documents to identify common thematic structures across the dataset. It allows you to specify the number of topics and the maximum number of features (words) to include in the model. Once the LDA model is trained, the module outputs two key results: a topic-word matrix (showing the most relevant words for each topic) and a document-topic matrix (indicating the topic distribution for each document). These outputs are saved to disk in CSV format and can be used for visualizations, labeling, or downstream trend detection. There is also a run_lda_timesliced.py version that runs this analysis separately on each time slice (e.g., monthly) to support topic evolution analysis over time.


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
