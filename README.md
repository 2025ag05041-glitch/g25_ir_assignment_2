# Text Mining & Document Intelligence System — IR Assignment 2 (Group G25)

A Streamlit application for a **technology knowledge library**. It loads 73 Wikipedia articles from five
practice areas, preprocesses the text, builds **manual TF-IDF** vectors and an **inverted index**,
classifies documents with a **manual Naive Bayes**, clusters them with a **manual K-Means**
(evaluated with manual **Purity** and **Rand Index**) and recommends related documents
(content-based, cosine similarity).

TF-IDF, the inverted index, Naive Bayes, K-Means, Purity and Rand Index are written from scratch
(`src/`). NumPy is used only for array arithmetic and NLTK only for the stop-word list, the Porter
stemmer and the WordNet lemmatizer. scikit-learn is not used anywhere (a unit test checks this).

## Project structure

```
app.py                     Streamlit entry point (page navigation)
config.py                  paths and default settings
views/                     the six pages: dashboard, dataset & upload, preprocessing,
                           text mining, recommendations, performance analysis
src/
  ingestion.py             loading files, doc IDs, metadata, uploads (.txt/.csv/.json)
  preprocessing.py         configurable cleaning, tokens, stop-words, stemming, lemmas, n-grams
  features.py              manual TF-IDF, L2 normalisation, cosine similarity
  inverted_index.py        postings lists + term-at-a-time search
  classification.py        manual multinomial Naive Bayes, stratified split, cross-validation
  clustering.py            manual K-Means, PCA (for the 2D plot only)
  evaluation.py            confusion matrix, accuracy/P/R/F1, Purity, Rand Index
  duplicates.py            exact / near-duplicate detection
  recommendation.py        Top-K recommendations
  experiments.py           experiments used in the report
  pipeline.py, app_state.py, visualization.py
Dataset/                   original text files (never modified)
data/metadata/             category_mapping.csv (manual labels + reason)
data/processed/            corpus_metadata.csv (generated: IDs, labels, source URLs)
reports/                   PDF report, results/ (csv/json), figures/, screenshots/
scripts/                   run_experiments.py, build_report.py, capture_screenshots.py,
                           make_submission.py
tests/                     pytest unit tests
evidence/                  BITS Virtual Lab screenshots and recording
G025_Contribution.xlsx     contribution of each member
```

## Setup

Tested with **Python 3.9.5** on Windows (any Python 3.9-3.12 should work).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4
```

The NLTK download is needed once. Without it the app still runs, using a built-in stop-word list and a
simple rule-based lemmatizer (the Preprocessing page shows which one is active), but the numbers will then
differ slightly from the report.

## Run the application

```bash
streamlit run app.py
```

Open http://localhost:8501. The dataset in `Dataset/` is loaded automatically.

## Using the application

| Page | What to do |
|---|---|
| Dashboard | Business context, corpus statistics, category distribution |
| Dataset & Upload | Browse/search/filter documents and read them; upload `.txt`, `.csv` or `.json` files (CSV/JSON need a `text` column, optional `title`, `category`, `doc_id`, `source`); duplicate analysis |
| Preprocessing & Features | Choose the preprocessing and TF-IDF options and press **Apply settings** (applies to all pages); before/after statistics, TF-IDF table, inverted index lookup, cosine similarity |
| Text Mining | Naive Bayes: split, α, actual vs predicted, metrics, classify new text. K-Means: K, initialisation, seeds, iterations, cluster membership |
| Recommendations | Select a document and K, see scores, categories and shared terms; or search with free text |
| Performance Analysis | Confusion matrix, class distributions, cluster scatter plot, cluster sizes, membership table, Purity / Rand Index vs NB metrics, experiments |

## Tests

```bash
python -m pytest -q
```

40 tests, including hand-calculated TF-IDF values and the textbook examples from Manning et al.
(Naive Bayes Example 13.1, Purity and Rand Index for Figure 16.4).

## Reproduce the results and the report

```bash
python scripts/run_experiments.py   # writes reports/results/ and reports/figures/ (~1 min)
python scripts/build_report.py      # writes reports/G25_IR_Assignment2_Report.pdf
```

Optional, to refresh the app screenshots used in the report (needs `pip install playwright` and
Google Chrome or `playwright install chromium`), with the app running on port 8501:

```bash
python scripts/capture_screenshots.py
```

## Demonstration (about 5 minutes)

1. Dashboard: explain the business context and the corpus (73 documents, 5 categories + 2 off-domain).
2. Dataset & Upload: search "neural", open a document; upload `Dataset/Stemming.txt` → Duplicate analysis
   shows an exact duplicate (UPL001 vs DOC061); remove the upload again.
3. Preprocessing: show before/after vocabulary (92,721 → 13,081), switch to lemmatization + bigrams,
   Apply, show the change; look up the term "network" in the inverted index; reset settings.
4. Text Mining: train Naive Bayes (accuracy 0.857), show wrong predictions; run K-Means K=5 and walk
   through seeds → iterations → stopping criterion → membership.
5. Recommendations: "Long short-term memory" → top-5 with scores and shared terms; text search.
6. Performance Analysis: confusion matrix, scatter plot, comparison tab, experiments tab.

## BITS Virtual Lab checklist (to be done by the group)

These cannot be produced outside the Virtual Lab and must not be faked:

- [ ] Copy the project folder to the Virtual Lab machine (or extract the ZIP there).
- [ ] Screenshot: terminal with `pip install -r requirements.txt` and `python -m pytest -q` (40 passed).
- [ ] Screenshot: terminal with `streamlit run app.py` running.
- [ ] Screenshots of every page in the Virtual Lab browser (Dashboard, Dataset & Upload, Preprocessing,
      Text Mining – NB and K-Means, Recommendations, Performance Analysis). Make the Virtual Lab
      desktop/URL visible in at least one screenshot.
- [ ] Short screen recording (3–5 min) following the demonstration steps above → `evidence/` (e.g.
      `G25_demo_recording.mp4`).
- [ ] Save the screenshots in `evidence/virtual_lab/` and run `python scripts/build_report.py`
      (they are added to Appendix A of the report).
- [ ] Run `python scripts/make_submission.py` and submit `G25_Assignment2.zip`.

## Notes

- The two off-domain articles (Carlos Lehder, Hippopotamuses in Colombia) are kept for browsing and
  recommendation but left out of classification/clustering evaluation.
- The real dataset has no duplicate documents. The effect of duplicates is shown with an in-memory
  experiment (Performance Analysis → Experiments). Dataset files are never changed.
- Uploaded documents are kept only for the current browser session.
