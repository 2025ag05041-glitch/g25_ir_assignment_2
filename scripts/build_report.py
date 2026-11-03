"""Build the PDF technical report from the saved experiment results.

Usage:
    python scripts/run_experiments.py      # creates reports/results + figures
    python scripts/build_report.py         # creates reports/G25_IR_Assignment2_Report.pdf

BITS Virtual Lab screenshots placed in evidence/virtual_lab/ (png/jpg) are
added to the report automatically the next time this script is run.
"""
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak,  # noqa: E402
                                Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

RESULTS = ROOT / "reports" / "results"
FIGURES = ROOT / "reports" / "figures"
SHOTS = ROOT / "reports" / "screenshots"
EVIDENCE = ROOT / "evidence" / "virtual_lab"
OUTPUT = ROOT / "reports" / "G25_IR_Assignment2_Report.pdf"

run = json.loads((RESULTS / "default_run.json").read_text())
dup = json.loads((RESULTS / "duplicate_experiment.json").read_text())
prep = pd.read_csv(RESULTS / "preprocessing_comparison.csv")
sweep = pd.read_csv(RESULTS / "k_sweep.csv")
team = pd.read_excel(ROOT / "G025_Contribution.xlsx", dtype={"Student Registration Number": str})

corpus = run["corpus"]
voc = run["vocabulary"]
clf = run["classification"]
m = clf["metrics"]
cv = clf["cv_metrics"]
clu = run["clustering"]
rec = run["recommendation"]
dups = run["duplicates"]
P = {row["setting"][0]: row for _, row in prep.iterrows()}   # rows keyed A..G

# --------------------------------------------------------------- self checks
# The discussion below states these facts; warn if a re-run changes them.
CLAIMS = {
    "cleaning shrinks vocabulary": P["B"]["vocabulary_size"] < P["A"]["vocabulary_size"] * 0.5,
    "stemming shrinks vocabulary more than lemmatization":
        P["D"]["vocabulary_size"] < P["E"]["vocabulary_size"] < P["C"]["vocabulary_size"],
    "bigrams give best purity and P@5": P["F"]["kmeans_purity"] == prep["kmeans_purity"].max()
        and P["F"]["recommendation_p_at_5"] == prep["recommendation_p_at_5"].max(),
    "no normalisation collapses k-means": P["G"]["kmeans_purity"] < 0.45,
    "minimal has highest NB CV accuracy": P["A"]["nb_cv_accuracy"] == prep["nb_cv_accuracy"].max(),
    "no duplicates in real corpus": dups["exact"] == 0 and dups["near"] == 0,
    "leakage inflates accuracy": dup["classification"]["with_duplicates_naive_split"]
        > dup["classification"]["with_duplicates_group_split"] + 0.03,
    "duplicates lower purity": dup["clustering"]["purity_with_duplicates"]
        < dup["clustering"]["purity_original"],
}
for claim, ok in CLAIMS.items():
    if not ok:
        print(f"WARNING: report text says '{claim}' but the results no longer show it")

# -------------------------------------------------------------------- styles
styles = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica", fontSize=10.2,
                      leading=14.2, spaceAfter=6)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.6, leading=11, textColor=colors.HexColor("#444444"))
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.4, leading=10.4, spaceAfter=0)
H1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=15,
                    spaceBefore=10, spaceAfter=8, textColor=colors.HexColor("#1a1a1a"))
H2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12,
                    spaceBefore=8, spaceAfter=5, textColor=colors.HexColor("#1a1a1a"))
FORMULA = ParagraphStyle("formula", parent=BODY, fontName="Courier", fontSize=9.2, leading=12.5,
                         leftIndent=18, spaceBefore=2, spaceAfter=8,
                         backColor=colors.HexColor("#f5f5f5"), borderPadding=5)
CAPTION = ParagraphStyle("caption", parent=SMALL, alignment=TA_CENTER, spaceAfter=10)
TITLE = ParagraphStyle("title", parent=H1, fontSize=20, leading=26, alignment=TA_CENTER, spaceAfter=6)
CENTER = ParagraphStyle("center", parent=BODY, alignment=TA_CENTER)

story = []
section_no = [0]
figure_no = [0]
table_no = [0]


def h1(text):
    section_no[0] += 1
    story.append(Paragraph(f"{section_no[0]}. {text}", H1))


def h2(text):
    story.append(Paragraph(text, H2))


def p(text, style=BODY):
    story.append(Paragraph(text, style))


def formula(text):
    story.append(Paragraph(text.replace("\n", "<br/>"), FORMULA))


def bullets(items):
    story.append(ListFlowable([ListItem(Paragraph(i, BODY), leftIndent=12) for i in items],
                              bulletType="bullet", start="•", leftIndent=14))


def table(rows, header=True, widths=None, caption=None, font=8.4):
    cell = ParagraphStyle("c", parent=CELL, fontSize=font, leading=font + 2)
    data = [[Paragraph(str(c), cell) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef5"))]
    t.setStyle(TableStyle(style))
    block = [t]
    if caption:
        table_no[0] += 1
        block.append(Paragraph(f"Table {table_no[0]}: {caption}", CAPTION))
    else:
        block.append(Spacer(1, 8))
    story.append(KeepTogether(block))


def figure(path, caption, width=16 * cm, max_height=21 * cm):
    if not Path(path).exists():
        p(f"<i>[missing figure: {Path(path).name}]</i>")
        return
    iw, ih = ImageReader(str(path)).getSize()
    w = width
    h = w * ih / iw
    if h > max_height:
        h = max_height
        w = h * iw / ih
    figure_no[0] += 1
    story.append(KeepTogether([Image(str(path), width=w, height=h),
                               Paragraph(f"Figure {figure_no[0]}: {caption}", CAPTION)]))


def f3(x):
    return f"{x:.3f}"


def pct(x):
    return f"{x * 100:.1f}%"


# ================================================================ title page
story += [Spacer(1, 3 * cm),
          Paragraph("Birla Institute of Technology and Science, Pilani", CENTER),
          Paragraph("Work Integrated Learning Programmes Division", CENTER),
          Spacer(1, 0.6 * cm),
          Paragraph("Information Retrieval (AIMLCZG537 / DSECLZG537)", CENTER),
          Paragraph("Assignment 2", CENTER),
          Spacer(1, 1.2 * cm),
          Paragraph("Text Mining and Document Intelligence System<br/>for a Technology Knowledge Library",
                    TITLE),
          Spacer(1, 0.4 * cm),
          Paragraph("Technical Report - Group G25", CENTER),
          Spacer(1, 1.5 * cm)]
rows = [["Student Registration Number", "Name"]]
rows += [[r["Student Registration Number"], r["Name"]] for _, r in team.iterrows()]
table(rows, widths=[6 * cm, 8 * cm])
story += [Spacer(1, 1.5 * cm), Paragraph(f"Date: {date.today().strftime('%d %B %Y')}", CENTER),
          PageBreak()]

# ================================================================ 1 intro
h1("Introduction")
p("This report describes the Text Mining and Document Intelligence System built by Group G25 for "
  "Assignment 2. The system is a Streamlit web application. It loads a collection of documents, "
  "preprocesses the text, builds TF-IDF features and an inverted index, classifies documents with Naive "
  "Bayes, groups them with K-Means, and recommends related documents. TF-IDF, the inverted index, "
  "Naive Bayes, K-Means, Purity and Rand Index were all implemented by us in Python, using NumPy only for "
  "basic array arithmetic. No scikit-learn or other library implementation of these methods is used. "
  "A unit test checks this automatically.")
p("All numbers in this report were produced by the code in the submission "
  "(<font face='Courier'>scripts/run_experiments.py</font>) with the default settings and random seed 42. "
  "The same values appear in the application under the same settings.")

# ================================================================ 2 business
h1("Selected Business Context")
p("We assume that our group is the text mining team of a technology consulting firm. The firm keeps an "
  "internal <b>knowledge library</b> of reference articles that consultants read before client "
  "engagements, for example a fintech analytics project or an NLP proof of concept. The library covers five "
  "practice areas: Machine Learning, Language &amp; Search Technology, AI Systems &amp; Cognition, Computing "
  "&amp; Infrastructure, and Finance &amp; Economics.")
p("The library is growing and is maintained by a small team. Analysts have three recurring needs:")
bullets(["<b>Tagging:</b> every new article must be assigned to a practice area so it can be found "
         "(document classification).",
         "<b>Topic discovery:</b> knowledge managers want to see which themes actually occur in the collection "
         "and whether the existing practice areas still fit (document clustering).",
         "<b>Related reading:</b> a consultant reading one article wants the most related articles "
         "(document recommendation)."])

# ================================================================ 3 problem
h1("Problem Statement")
p("Given a collection of unstructured, noisy text documents with category labels, build an interactive "
  "system that (1) ingests and explores the documents and detects duplicates, (2) preprocesses the text in "
  "a configurable way and represents documents with manually computed, length-normalised TF-IDF vectors and "
  "an inverted index, (3) classifies documents with a manually implemented Naive Bayes classifier, (4) "
  "clusters documents with a manually implemented K-Means algorithm evaluated by Purity and Rand Index, (5) "
  "recommends the Top-K related documents, and (6) visualises and compares the results.")

# ================================================================ 4 dataset
h1("Dataset Description")
p(f"The dataset is the folder <font face='Courier'>Dataset/</font> with <b>{corpus['documents']} plain-text "
  f"files</b>, each an English Wikipedia article exported as text. Together they contain "
  f"<b>{corpus['total_words']:,} words</b> ({corpus['total_mb']:.2f} MB of text after the source footer is "
  f"removed). The assignment asks for at least 50-75 documents <i>or</i> about 5 MB of raw text. Our "
  f"{corpus['documents']} documents meet the document-count condition. The text size is below 5 MB, but "
  "each document is long (on average {0:,.0f} words), so the corpus is large enough for meaningful TF-IDF "
  "statistics.".format(corpus["avg_words"]))
p("The text is realistic and noisy. Because it was exported from formatted web pages, it contains "
  "infobox fragments mixed into sentences, broken lines, URLs, citation markers such as [12], long "
  "reference lists with author names and journal titles, mixed casing and many domain-specific terms. "
  "Document length varies strongly, from "
  f"{corpus['min_words']} words (<i>Tokenization</i>, a short disambiguation page) to "
  f"{corpus['max_words']:,} words (<i>Deep learning</i>).")
h2("Document structure and metadata")
p("Text and metadata are kept separate. The raw text stays unchanged in <font face='Courier'>Dataset/</font>. "
  "The metadata is generated by <font face='Courier'>src/ingestion.py</font> and saved in "
  "<font face='Courier'>data/processed/corpus_metadata.csv</font>:")
table([["Field", "Description"],
       ["doc_id", "Unique, stable ID (DOC001-DOC073, in file-name order; uploads get UPL001, ...)"],
       ["title", "Article title (first line of the file)"],
       ["category", "Practice-area label from data/metadata/category_mapping.csv"],
       ["file_name", "Original file in Dataset/ (traceability to the raw document)"],
       ["source / source_url / revision_id", "Wikipedia, and the exact revision URL taken from the "
                                             "'Retrieved from' footer (the footer is removed from the text)"],
       ["char_count / word_count", "Length statistics"],
       ["text", "Main text content (in memory; read from the original file)"]],
      widths=[4.5 * cm, 11.5 * cm], caption="Document fields")
h2("Category labels")
p("The files did not come with labels, so we labelled them manually. We used the topic of each article's "
  "lead section, not only its file name. The mapping and a one-line reason for every document are in "
  "<font face='Courier'>data/metadata/category_mapping.csv</font>. Two articles (<i>Carlos Lehder</i>, a "
  "biography, and <i>Hippopotamuses in Colombia</i>) do not belong to any practice area. They are labelled "
  "<i>General (Off-domain)</i>. They stay in the corpus for browsing and recommendation, but are left out of "
  "classification and clustering evaluation, because two documents are too few to train and test a class. "
  f"This leaves {sum(v for k, v in run['category_counts'].items() if 'Off-domain' not in k)} labelled "
  "documents in five classes.")
rows = [["Category", "Documents", "Examples"]]
examples = {"Machine Learning": "Backpropagation, Random forest, LSTM, GAN",
            "Language & Search Technology": "Stemming, Search engine, Machine translation",
            "Finance & Economics": "Stock market, Financial crisis, Layoff",
            "AI Systems & Cognition": "ChatGPT, Turing test, Intelligent agent",
            "Computing & Infrastructure": "Cloud computing, Data center, Cyberattack",
            "General (Off-domain)": "Carlos Lehder, Hippopotamuses in Colombia"}
for c, n in run["category_counts"].items():
    rows.append([c.replace("&", "&amp;"), n, examples.get(c, "")])
table(rows, widths=[5.5 * cm, 2.2 * cm, 8.3 * cm], caption="Category distribution")
figure(FIGURES / "category_distribution.png", "Documents per category", width=12 * cm)
p("Some labels are debatable, and that is realistic. <i>Goldmines Telefilms</i> is a company profile of "
  "a film distributor that we put under Finance &amp; Economics (business). <i>Conversation</i> and "
  "<i>Human intelligence</i> were put under AI Systems &amp; Cognition. These borderline cases are visible "
  "later in the classification and clustering errors.")
figure(FIGURES / "length_distribution.png", "Distribution of document lengths (words)", width=12 * cm)

# ================================================================ 5 duplicates
h1("Dataset Preparation and Duplicate Analysis")
p("Ingestion reads every file as UTF-8 (removing the byte-order mark), normalises line endings, takes the "
  "first line as the title, and moves the <i>Retrieved from</i> footer into the metadata. Empty uploaded "
  "documents are skipped with a warning. Uploaded documents (.txt, .csv or .json) get unique IDs and are "
  "merged with the base corpus for the current session.")
p("Duplicates are detected in two ways (<font face='Courier'>src/duplicates.py</font>):")
bullets(["<b>Exact duplicates</b>: equal MD5 hash of the text after lowercasing and removing punctuation "
         "and extra whitespace.",
         "<b>Near duplicates</b>: Jaccard similarity of the sets of word 5-gram shingles, "
         "J(A,B) = |A &#8745; B| / |A &#8746; B|, and the cosine similarity of the TF-IDF vectors. A pair is "
         "flagged if J &#8805; 0.5 or cosine &#8805; 0.9."])
p(f"All {dups['pairs_compared']:,} document pairs of the real corpus were compared. <b>No exact or near "
  f"duplicates were found.</b> The most similar pair is <i>{dups['top_pairs'][0]['title_a']}</i> / "
  f"<i>{dups['top_pairs'][0]['title_b']}</i> with Jaccard {f3(dups['top_pairs'][0]['jaccard'])} and cosine "
  f"{f3(dups['top_pairs'][0]['cosine'])}. These are related articles, not copies. Table 3 lists the closest pairs.")
rows = [["Document A", "Document B", "Jaccard (5-shingles)", "TF-IDF cosine"]]
for r in dups["top_pairs"][:6]:
    rows.append([r["title_a"], r["title_b"], f3(r["jaccard"]), f3(r["cosine"])])
table(rows, widths=[5.2 * cm, 5.2 * cm, 2.8 * cm, 2.8 * cm], caption="Most similar document pairs in the corpus")
p("Because the corpus has no duplicates, we did <b>not</b> add copies to the dataset files. Instead, (a) "
  "the application lets an analyst upload a file and see it flagged (Figure 3: re-uploading "
  "<i>Stemming.txt</i> is detected as an exact duplicate of DOC061), and (b) a separate, clearly labelled "
  "in-memory experiment measures the effect of duplicates (Section 16, Question 2).")
p("<b>Handling.</b> Flagged documents form a duplicate group (union-find). The train/test split and "
  "cross-validation keep every group on one side, so a copy of a test document can never be in the training "
  "data. The recommender hides documents above the duplicate threshold, and the original files are never "
  "modified.")
figure(SHOTS / "13_upload_duplicate_detected.png",
       "Duplicate analysis after uploading a copy of Stemming.txt (local run of the application)",
       width=14 * cm)

# ================================================================ 6 architecture
h1("System Architecture")
p("The project is split into small modules. The Streamlit pages only handle the user interface and call "
  "the functions in <font face='Courier'>src/</font>. The experiment script calls the same functions, so "
  "the report and the application always show the same numbers.")
formula("Dataset/*.txt + uploads --> ingestion --> preprocessing --> TF-IDF + inverted index\n"
        "        --> Naive Bayes classification  --> metrics (accuracy, P, R, F1, confusion matrix)\n"
        "        --> K-Means clustering          --> Purity, Rand Index, 2D PCA plot\n"
        "        --> cosine similarity           --> Top-K recommendations / text search\n"
        "        --> Streamlit pages (views/)    and   scripts/run_experiments.py -> report")
table([["Module", "Responsibility"],
       ["src/ingestion.py", "Load files, IDs, metadata, uploads (.txt/.csv/.json), corpus statistics"],
       ["src/preprocessing.py", "Configurable cleaning, tokenization, stop-words, stemming, lemmatization, n-grams"],
       ["src/features.py", "Manual vocabulary, tf, df, idf, TF-IDF, L2 normalisation, cosine similarity"],
       ["src/inverted_index.py", "Postings lists and term-at-a-time cosine search"],
       ["src/classification.py", "Manual multinomial Naive Bayes, stratified split, cross-validation"],
       ["src/clustering.py", "Manual K-Means (k-means++/random init), PCA projection for plotting only"],
       ["src/evaluation.py", "Manual confusion matrix, accuracy/P/R/F1, Purity, Rand Index"],
       ["src/duplicates.py", "Exact and near-duplicate detection, duplicate groups"],
       ["src/recommendation.py", "Top-K content-based recommendation, explanation terms, precision@K"],
       ["src/experiments.py", "Preprocessing comparison, duplicate experiment, K sweep"],
       ["src/app_state.py, views/, app.py", "Streamlit session state, caching and the six pages"],
       ["tests/", "40 unit tests (pytest) including textbook examples"]],
      widths=[5 * cm, 11 * cm], caption="Project modules")
p("The application has six pages: <b>Dashboard</b> (business context and corpus overview), <b>Dataset &amp; "
  "Upload</b> (browse, search, filter, upload, duplicate analysis), <b>Preprocessing &amp; Features</b> "
  "(settings, before/after statistics, TF-IDF, inverted index, cosine similarity), <b>Text Mining</b> (Naive "
  "Bayes and K-Means with all algorithm steps), <b>Recommendations</b> and <b>Performance Analysis</b> "
  "(charts, comparison and experiments). Results are cached with a key built from a fingerprint of the "
  "corpus and all settings. Changing a setting or uploading a document recomputes the results, and unchanged "
  "settings reuse the cache.")
figure(SHOTS / "01_dashboard.png", "Dashboard page (local run)", width=14 * cm)

# ================================================================ 7 preprocessing
h1("Text Preprocessing")
p("The preprocessing pipeline (<font face='Courier'>src/preprocessing.py</font>) is fully configurable from "
  "the Preprocessing page and applies its steps in this order:")
bullets(["<b>Noise cleaning</b> (optional): remove the References / External links section, URLs and "
         "citation markers such as [12].",
         "<b>Lowercasing</b> (optional).",
         "<b>Tokenization</b>: a regular-expression tokenizer that keeps runs of letters/digits (default), or a "
         "plain whitespace split for comparison.",
         "<b>Filtering</b>: stop-word removal (NLTK English list plus a few wiki words such as 'retrieved', "
         "'archived'), removal of tokens containing digits, minimum token length 2.",
         "<b>Word normalisation</b>: none, Porter stemming (NLTK), or WordNet lemmatization (NLTK; a small "
         "rule-based fallback is used if WordNet data is not installed).",
         "<b>N-grams</b>: unigrams, unigrams + bigrams, or up to trigrams (joined with '_', built from the "
         "filtered tokens)."])
p("The default setting is: lowercase, regex tokens, noise and references removed, stop-words and numbers "
  "removed, Porter stemming, unigrams. Table 5 and Figures 5-6 compare the corpus before preprocessing "
  "(plain whitespace tokens of the raw text) and after the default pipeline.")
table([["Statistic", "Before", "After (default)", "Change"],
       ["Vocabulary size", f"{voc['raw_vocabulary']:,}", f"{voc['processed_vocabulary']:,}",
        pct(voc['processed_vocabulary'] / voc['raw_vocabulary'] - 1)],
       ["Total tokens", f"{voc['raw_tokens']:,}", f"{voc['processed_tokens']:,}",
        pct(voc['processed_tokens'] / voc['raw_tokens'] - 1)],
       ["Average tokens per document", f"{voc['raw_avg_doc_length']:,.0f}",
        f"{voc['processed_avg_doc_length']:,.0f}", pct(voc['processed_avg_doc_length'] / voc['raw_avg_doc_length'] - 1)],
       ["TF-IDF features (df >= 2)", "-", f"{voc['tfidf_features']:,}", ""]],
      widths=[6 * cm, 3 * cm, 3.5 * cm, 3.5 * cm], caption="Vocabulary before and after preprocessing")
figure(FIGURES / "top_terms_before_after.png",
       "Most frequent terms before and after preprocessing. Before, the list is dominated by stop-words and "
       "wiki noise ('Retrieved', 'Archived', 'original'). After, it shows domain terms.", width=15 * cm)
figure(FIGURES / "rank_frequency.png", "Rank-frequency (Zipf) curves before and after preprocessing",
       width=12 * cm)
p("Before preprocessing the most frequent terms are <i>"
  + ", ".join(t for t, _ in voc["top_raw_terms"][:6]) + "</i>. After preprocessing they are <i>"
  + ", ".join(t for t, _ in voc["top_processed_terms"][:8]) + "</i>. Very general words such as 'use', "
  "'one' and 'also' remain, because they are not in the NLTK stop-word list. However, they occur in "
  "almost every document, so their IDF, and therefore their TF-IDF weight, is close to zero.")

# ================================================================ 8 tfidf
h1("Manual TF-IDF and Inverted Index")
p("TF-IDF is implemented from scratch in <font face='Courier'>src/features.py</font>. For N documents:")
formula("tf(t,d)    = number of times term t occurs in document d\n"
        "df(t)      = number of documents that contain t\n"
        "w_tf(t,d)  = 1 + log10(tf(t,d))   if tf > 0   (option: raw tf)\n"
        "idf(t)     = log10(N / df(t))\n"
        "w(t,d)     = w_tf(t,d) * idf(t)\n"
        "w'(t,d)    = w(t,d) / sqrt( sum_t' w(t',d)^2 )      (L2 length normalisation)\n"
        "cos(d1,d2) = sum_t w'(t,d1) * w'(t,d2)")
p("We use the logarithmic tf so that a word repeated 100 times does not count 100 times more than a word "
  "that appears once. A term that appears in every document gets idf = 0 and no weight. The vocabulary "
  "contains only terms with df &#8805; 2 (configurable), which removes most typos and one-off tokens from "
  "the reference lists. Documents are stored as sparse dictionaries {term: weight} and also as a dense "
  "NumPy matrix for K-Means. L2 normalisation makes every vector unit length, so long and short documents "
  "can be compared fairly, and cosine similarity becomes a dot product.")
p(f"<b>Inverted index.</b> For every term the index stores a postings list of (document, raw tf, TF-IDF "
  f"weight), sorted by document (<font face='Courier'>src/inverted_index.py</font>). With the default "
  f"settings it has {voc['index']['terms']:,} terms and {voc['index']['postings']:,} postings (average "
  f"postings length {voc['index']['avg_postings_length']:.1f}, longest {voc['index']['max_postings_length']}). "
  "Free-text queries are answered term-at-a-time. For each query term we walk its postings list and add "
  "query weight x document weight to the document's score. Because both vectors are normalised, the score "
  "is exactly the cosine similarity, and only documents that share at least one term with the query are "
  "visited. A unit test checks that index scores equal the direct cosine computation.")
figure(SHOTS / "04_preprocessing.png", "Preprocessing page with settings and before/after statistics (local run)",
       width=13 * cm, max_height=18 * cm)

# ================================================================ 9 naive bayes
h1("Naive Bayes Classification")
p("We implemented a multinomial Naive Bayes classifier with Laplace (add-alpha) smoothing "
  "(<font face='Courier'>src/classification.py</font>). It is trained on the term counts of the "
  "preprocessed documents:")
formula("P(c)       = N_c / N\n"
        "P(t | c)   = (count(t,c) + alpha) / (sum_t' count(t',c) + alpha * |V|)\n"
        "score(c|d) = log P(c) + sum_{t in d} tf(t,d) * log P(t|c)\n"
        "prediction = argmax_c score(c|d)")
p("Logarithms prevent numerical underflow when thousands of small probabilities are multiplied. The "
  "vocabulary V, the priors and the likelihoods are computed from the <b>training documents only</b>. "
  "Words of a test document that never appeared in training are ignored, so the test set cannot influence "
  "the model. The implementation reproduces the worked example 13.1 of Manning et al. (2008) exactly "
  "(unit test).")
p(f"<b>Data split.</b> The {clf['train_docs'] + clf['test_docs']} labelled documents are split 70/30 with a "
  f"stratified split (seed 42), giving {clf['train_docs']} training and {clf['test_docs']} test documents "
  "with every class present in both sets. Duplicate groups are kept on one side of the split. Because the "
  "test set is small, we also report stratified 5-fold cross-validation, in which every document is tested "
  "exactly once.")
p("<b>Metrics</b> (implemented in <font face='Courier'>src/evaluation.py</font>): accuracy, and per class "
  "precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R). We report the macro average (every class "
  "counts equally, which matters because our classes have different sizes) and the weighted average "
  "(weighted by class support).")

# ================================================================ 10 kmeans
h1("K-Means Clustering")
p("K-Means is implemented from scratch in <font face='Courier'>src/clustering.py</font> and runs on the "
  "L2-normalised TF-IDF vectors of the 71 labelled documents. The steps shown in the application are:")
bullets(["<b>Initialisation</b>: k-means++ (default). The first centroid is a random document, and each next "
         "centroid is a document chosen with probability proportional to its squared distance from the "
         "nearest existing centroid. Plain random initialisation is also available.",
         "<b>Assignment</b>: each document goes to the centroid with the smallest squared Euclidean distance. "
         "For unit-length vectors ||x - c||<super>2</super> = 1 + ||c||<super>2</super> - 2 x.c, so this "
         "ranks centroids by cosine-like similarity.",
         "<b>Update</b>: each centroid becomes the mean vector of its documents. An empty cluster is "
         "re-seeded with the document that is farthest from its current centroid.",
         "<b>Stopping criterion</b>: stop when no document changes cluster or the largest centroid "
         "movement is below 10<super>-6</super>, or after the maximum number of iterations (100).",
         "The algorithm is run 10 times with different seeds, and the run with the lowest SSE "
         "(within-cluster sum of squared distances) is kept."])
p(f"With K = {clu['k']} (the number of true categories), the best run started from the seed documents "
  f"<i>{', '.join(clu['initial_seeds'])}</i>. It converged after {clu['n_iter']} iterations "
  f"(SSE {clu['sse']:.3f}). All 10 runs converged within 2-3 iterations. TF-IDF vectors are very "
  "high-dimensional and sparse, and documents are almost orthogonal to documents of other topics, so "
  "k-means++ seeds usually lead to a stable assignment after the first update.")
figure(SHOTS / "06_kmeans.png", "K-Means page: initial seeds, iterations, cluster sizes and descriptions (local run)",
       width=13 * cm, max_height=19 * cm)

# ================================================================ 11 cluster eval
h1("Clustering Evaluation")
p("Purity and Rand Index are implemented without any library (<font face='Courier'>src/evaluation.py</font>):")
formula("Purity = (1/N) * sum_k max_j |cluster_k  intersect  class_j|\n"
        "RI     = (TP + TN) / (TP + FP + FN + TN)\n"
        "  TP: pair in same cluster and same class    FP: same cluster, different class\n"
        "  FN: different cluster, same class          TN: different cluster and class")
p("The pair counts are computed from the cluster-class contingency table: "
  "TP + FP = &#931;<sub>k</sub> C(n<sub>k</sub>, 2) and TP = &#931;<sub>k,j</sub> C(n<sub>kj</sub>, 2). "
  "A second brute-force version that loops over all pairs is used in the unit tests to check the fast "
  "version. Both metrics reproduce the textbook example of Manning et al. (Figure 16.4: purity 0.71, "
  "RI 0.68). Purity measures how 'clean' each cluster is. Rand Index measures agreement on all document "
  "pairs. Purity always increases as K grows (with K = N it is 1), so it must be compared at a fixed K.")

# ================================================================ 12 recommendation
h1("Document Recommendation")
p("We implemented <b>content-based</b> recommendation (<font face='Courier'>src/recommendation.py</font>). "
  "For a selected document d, every other document d' is scored by the cosine similarity of their "
  "TF-IDF vectors, and the K highest are returned. The selected document is always excluded. Documents "
  "with cosine &#8805; 0.9 can be hidden as near-duplicates. For each recommendation the application shows "
  "the document ID, title, category, similarity score, whether the category matches, and the "
  "<i>shared terms</i>: the terms with the largest product w(t,d)&#183;w(t,d'), which explain the score. "
  "Users can also type free text, which is scored through the inverted index.")
p(f"As a simple offline quality measure we use <b>category precision@5</b>, the share of the top-5 "
  f"recommendations that have the same category as the selected document, averaged over all labelled "
  f"documents. With the default settings it is <b>{f3(rec['p_at_5'])}</b>.")

# ================================================================ 13 results
h1("Experimental Results")
h2("13.1 Naive Bayes")
table([["Metric", "Test split (21 docs)", "5-fold CV (71 docs)"],
       ["Accuracy", f3(m["accuracy"]), f3(cv["accuracy"])],
       ["Macro precision", f3(m["macro_precision"]), f3(cv["macro_precision"])],
       ["Macro recall", f3(m["macro_recall"]), f3(cv["macro_recall"])],
       ["Macro F1", f3(m["macro_f1"]), f3(cv["macro_f1"])],
       ["Weighted F1", f3(m["weighted_f1"]), f3(cv["weighted_f1"])]],
      widths=[5 * cm, 5 * cm, 5 * cm], caption="Naive Bayes evaluation (default settings, alpha = 1)")
rows = [["Class", "Precision", "Recall", "F1", "Support"]]
for c in m["per_class"]:
    rows.append([c["class"].replace("&", "&amp;"), f3(c["precision"]), f3(c["recall"]), f3(c["f1"]),
                 c["support"]])
table(rows, widths=[6 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm], caption="Per-class metrics on the test split")
wrong = [x for x in clf["predictions"] if x["actual"] != x["predicted"]]
rows = [["Doc ID", "Title", "Actual", "Predicted", ""]]
for x in clf["predictions"]:
    rows.append([x["doc_id"], x["title"], x["actual"].replace("&", "&amp;"),
                 x["predicted"].replace("&", "&amp;"), "" if x["actual"] == x["predicted"] else "wrong"])
table(rows, widths=[1.8 * cm, 5.2 * cm, 3.9 * cm, 3.9 * cm, 1.2 * cm],
      caption="Actual and predicted labels of all test documents", font=7.6)
p(f"{len(wrong)} of {clf['test_docs']} test documents are misclassified: "
  + "; ".join(f"<i>{x['title']}</i> ({x['actual'].replace('&', '&amp;')} predicted as "
              f"{x['predicted'].replace('&', '&amp;')})" for x in wrong)
  + ". Each error makes sense. <i>Neural machine translation</i> is mostly about neural network "
  "architectures, <i>Data structure</i> uses algorithmic vocabulary that is common in the ML articles, and "
  "<i>Representative agent</i> shares the word 'agent' with the AI agent articles. Machine Learning, the "
  "largest class, has perfect recall but lower precision, because borderline documents are pulled into it.")
figure(FIGURES / "actual_vs_predicted.png", "Actual vs predicted class distribution on the test set", width=12 * cm)
figure(FIGURES / "confusion_matrix.png",
       "Confusion matrices: test split (left) and pooled 5-fold cross-validation (right)", width=15 * cm)
figure(FIGURES / "per_class_metrics.png", "Precision, recall and F1 per class (test split)", width=12 * cm)
p("In cross-validation the weakest classes are <b>Finance &amp; Economics</b> (recall "
  f"{cv['per_class'][2]['recall']:.2f}) and <b>Computing &amp; Infrastructure</b> (recall "
  f"{cv['per_class'][1]['recall']:.2f}). These are the smallest and most heterogeneous classes. For "
  "example, Finance contains a film company profile and an article on cryptocurrency crime. The most "
  "indicative Naive Bayes terms show this too. For Finance they include "
  f"<i>{', '.join(clf['top_terms']['Finance & Economics'][:8])}</i>, where 'telugu', 'tamil' and 'bhojpuri' "
  "come only from the Goldmines Telefilms article.")
h2("13.2 K-Means")
rows = [["Cluster", "Size", "Composition (true categories)", "Top centroid terms"]]
for c in sorted(clu["contingency"], key=int):
    comp = ", ".join(f"{k.split(' ')[0]} {v}" for k, v in sorted(clu["contingency"][c].items(),
                                                                key=lambda kv: -kv[1]))
    rows.append([c, clu["sizes"][int(c)], comp, ", ".join(clu["top_terms"][int(c)][:6])])
table(rows, widths=[1.5 * cm, 1.3 * cm, 6.2 * cm, 7 * cm], caption=f"K-Means clusters (K = {clu['k']})")
pc = clu["pair_counts"]
p(f"<b>Purity = {f3(clu['purity'])}</b> and <b>Rand Index = {f3(clu['rand_index'])}</b> "
  f"(TP {pc['TP']}, FP {pc['FP']}, FN {pc['FN']}, TN {pc['TN']} over {sum(pc.values()):,} pairs). Some "
  "clusters are clean. The economics-theory cluster is 100% Finance, and the cluster about philosophy of "
  "mind and the Turing test is mostly AI Systems. Others mix categories. The largest cluster contains the "
  "statistical / ML articles together with 8 language-technology articles that use the same "
  "vocabulary (vectors, models, training). Another cluster collects web-, service- and security-oriented "
  "articles from four categories. K-Means groups documents by shared vocabulary, which does not always match "
  "our human-defined practice areas.")
figure(FIGURES / "cluster_scatter.png",
       "2D PCA projection of the TF-IDF vectors, coloured by K-Means cluster (left) and true category "
       f"(right). PCA is used only for plotting (2 components explain "
       f"{sum(clu['explained_variance_2d']) * 100:.1f}% of the variance).", width=16 * cm)
figure(FIGURES / "cluster_sizes_contingency.png", "Cluster sizes and cluster vs category contingency table",
       width=16 * cm)
p("<b>Effect of K and initialisation.</b> Table 10 shows Purity, Rand Index and SSE for K = 2-10. SSE "
  "decreases steadily and has no clear elbow. Purity generally rises with K because smaller clusters are "
  "purer, but it is not monotonic because each K is a separate local optimum. Running the same "
  "configuration with five different seeds gave a mean purity of "
  f"{f3(P['D']['kmeans_purity'])} and mean RI of {f3(P['D']['kmeans_rand_index'])} at K = 5. Random "
  "initialisation reached purity 0.690 / RI 0.820, even though its best SSE was higher. A lower SSE does not "
  "guarantee better agreement with the human labels.")
rows = [["K", "Purity", "Rand Index", "SSE"]] + [[int(r.k), f3(r.purity), f3(r.rand_index), f"{r.sse:.2f}"]
                                                 for r in sweep.itertuples()]
table(rows, widths=[2 * cm, 3 * cm, 3 * cm, 3 * cm], caption="K-Means results for different K (seed 42)")
figure(FIGURES / "k_sweep.png", "Purity, Rand Index and SSE for different K", width=14 * cm)

h2("13.3 Recommendations")
for title in ["Long short-term memory", "Stock market", "Agent (economics)"]:
    ex = rec["examples"][title]
    rows = [["Rank", "Doc ID", "Title", "Category", "Cosine", "Shared terms"]]
    for r in ex["recommendations"]:
        rows.append([r["rank"], r["doc_id"], r["title"], r["category"].replace("&", "&amp;"),
                     f3(r["similarity"]), r["shared_terms"]])
    table(rows, widths=[1.1 * cm, 1.6 * cm, 3.8 * cm, 3.3 * cm, 1.4 * cm, 4.8 * cm],
          caption=f"Top-5 recommendations for '{title}' ({ex['doc_id']}, {ex['category'].replace('&', '&amp;')})",
          font=7.8)
p("The recommendations are mostly sensible. <i>Long short-term memory</i> leads to recurrent networks, deep "
  "learning and speech recognition, which uses LSTMs. <i>Stock market</i> leads to the other finance "
  "articles. Two weaknesses are also visible. For <i>Stock market</i>, the 5th result is <i>Data center</i>, "
  "matched on weak terms such as 'floor', 'tax' and 'ownership'. For <i>Agent (economics)</i>, "
  "<i>Software agent</i> appears because of the ambiguous word 'agent'. "
  f"Free-text search for \"{rec['query']}\" returned "
  + ", ".join(f"<i>{r['title']}</i> ({f3(r['similarity'])})" for r in rec["query_results"][:5]) + ".")
figure(SHOTS / "07_recommendations.png", "Recommendation interface (local run)", width=13 * cm, max_height=15 * cm)

# ================================================================ 14 visualisations
h1("Visualizations")
p("The application contains all required visualisations. The <b>Performance Analysis</b> page shows the "
  "actual vs predicted class distribution, the confusion matrices (test split and cross-validation), and the "
  "per-class metrics. Its Clustering tab shows the 2D scatter plot (switchable between cluster colours and "
  "true-category colours), the cluster-size bar chart, the cluster vs category heatmap and the cluster "
  "membership table (document ID, title, true label, assigned cluster). The Comparison tab shows "
  "Purity and Rand Index next to the Naive Bayes metrics with an interpretation, and the Experiments tab "
  "shows the experiments below. The Preprocessing page shows the term-frequency charts, the Zipf curves, the "
  "document-frequency histogram and a document-similarity heatmap. The Recommendations page shows the score "
  "chart.")
figure(SHOTS / "10_perf_clustering.png", "Clustering visualisations on the Performance Analysis page (local run)",
       width=13 * cm, max_height=17 * cm)

# ================================================================ 15 comparison
h1("Comparative Analysis")
figure(FIGURES / "nb_vs_kmeans.png", "Naive Bayes metrics (test split) next to K-Means Purity and Rand Index",
       width=12 * cm)
table([["Aspect", "Naive Bayes", "K-Means"],
       ["Learning type", "Supervised: uses the category labels", "Unsupervised: never sees the labels"],
       ["Output", "One of the 5 known categories per document", "K groups without names"],
       ["Evaluation", "Accuracy, precision, recall, F1, confusion matrix",
        "Purity, Rand Index (labels used only for evaluation)"],
       ["Result (default)", f"Accuracy {f3(m['accuracy'])}, macro F1 {f3(m['macro_f1'])} "
                            f"(CV: {f3(cv['accuracy'])} / {f3(cv['macro_f1'])})",
        f"Purity {f3(clu['purity'])}, RI {f3(clu['rand_index'])} "
        f"(5-seed mean {f3(P['D']['kmeans_purity'])} / {f3(P['D']['kmeans_rand_index'])})"],
       ["Main errors", "Cross-domain documents pulled into the largest class",
        "Clusters follow shared vocabulary (e.g. web/security, sequence models) rather than practice areas"],
       ["Use in the library", "Auto-tagging new articles", "Discovering themes, checking the category scheme"]],
      widths=[3.2 * cm, 6.4 * cm, 6.4 * cm], caption="Naive Bayes vs K-Means")
p("The two methods solve different problems, so their numbers cannot be compared directly. Accuracy "
  "checks whether each document received the correct label. Purity checks whether each cluster is dominated "
  "by one label, and Rand Index checks pairwise agreement. Naive Bayes clearly fits our labels better, "
  "because it learns them. K-Means reaches a purity of about 0.6-0.65 without any labels. That is useful, "
  "but not reliable enough to replace manual tagging.")

# ================================================================ 16 inference
h1("Compulsory Inference and Discussion")
h2("Question 1: How did the choice of preprocessing steps affect the vocabulary, classification, "
   "clustering and recommendations?")
p("We evaluated seven preprocessing settings. Each setting changes one step compared with an earlier "
  "setting, as its name shows. Every setting was run on the same corpus with the same split and seeds "
  "(Table 15, Figure 18).")
rows = [["Setting", "Vocab", "TF-IDF feat.", "Terms/doc", "NB acc (test)", "NB acc (CV)", "NB F1 (CV)",
         "Purity*", "RI*", "P@5"]]
for _, r in prep.iterrows():
    rows.append([r["setting"], f"{r['vocabulary_size']:,}", f"{r['tfidf_features']:,}",
                 f"{r['avg_terms_per_doc']:,.0f}", f3(r["nb_accuracy"]), f3(r["nb_cv_accuracy"]),
                 f3(r["nb_cv_macro_f1"]), f3(r["kmeans_purity"]), f3(r["kmeans_rand_index"]),
                 f3(r["recommendation_p_at_5"])])
table(rows, widths=[4.2 * cm, 1.4 * cm, 1.4 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 1.2 * cm,
                    1.2 * cm, 1.1 * cm],
      caption="Effect of preprocessing (* K-Means K = 5, mean over 5 seeds; P@5 = category precision@5)",
      font=7.2)
figure(FIGURES / "preprocessing_comparison.png", "Effect of preprocessing settings on the three tasks",
       width=15 * cm)
bullets([
    f"<b>Vocabulary.</b> Cleaning (removing reference sections, URLs and numbers) has the largest effect. It "
    f"cuts the vocabulary from {P['A']['vocabulary_size']:,} to {P['B']['vocabulary_size']:,} terms and halves "
    f"the terms per document ({P['A']['avg_terms_per_doc']:,.0f} to {P['B']['avg_terms_per_doc']:,.0f}), so "
    "about half of the raw text is reference lists. Stop-word removal hardly changes the vocabulary (only "
    f"~150 words) but removes 36% of all tokens. Stemming reduces the vocabulary to "
    f"{P['D']['vocabulary_size']:,} (-31%), and lemmatization is less aggressive "
    f"({P['E']['vocabulary_size']:,}). Adding bigrams multiplies the vocabulary to "
    f"{P['F']['vocabulary_size']:,}.",
    f"<b>Classification.</b> Naive Bayes was robust. Cross-validated accuracy stayed between "
    f"{prep['nb_cv_accuracy'].min():.3f} and {prep['nb_cv_accuracy'].max():.3f}, and on the 21-document test "
    "split the settings differ by only one document. Surprisingly, the minimal setting had the highest CV "
    f"accuracy ({f3(P['A']['nb_cv_accuracy'])}). Naive Bayes can use the noisy tokens: reference "
    "lists contain names of authors, journals and conferences that are strongly topic-specific. "
    "Stemming and lemmatization did not change accuracy, and bigrams slightly reduced it "
    f"({f3(P['F']['nb_cv_accuracy'])}). Bigrams add many rare features that make the probability estimates "
    "noisier for a training set of only ~57 documents per fold.",
    f"<b>Clustering</b> was more sensitive. Cleaning improved purity from {f3(P['A']['kmeans_purity'])} to "
    f"{f3(P['B']['kmeans_purity'])} and RI from {f3(P['A']['kmeans_rand_index'])} to "
    f"{f3(P['B']['kmeans_rand_index'])}. Without cleaning, the shared citation vocabulary makes unrelated "
    f"documents look similar. Bigrams gave the best clusters (purity {f3(P['F']['kmeans_purity'])}, RI "
    f"{f3(P['F']['kmeans_rand_index'])}) because phrases like 'neural_network' or 'stock_market' are more "
    "topic-specific than single words. The most important factor was <b>length normalisation</b>. Without "
    f"it, purity fell to {f3(P['G']['kmeans_purity'])} and RI to {f3(P['G']['kmeans_rand_index'])}. With "
    "the seed-42 run, 57 of 71 documents ended in one cluster while the three longest articles formed a "
    "cluster of their own: Euclidean distance then measures document length, not topic.",
    f"<b>Recommendations</b> improved steadily with cleaning, stemming/lemmatization and bigrams (P@5 from "
    f"{f3(P['A']['recommendation_p_at_5'])} to {f3(P['F']['recommendation_p_at_5'])}). Normalisation does "
    "not change recommendations (setting G = D), because cosine similarity already divides by the vector "
    "lengths.",
    "<b>Conclusion:</b> noise cleaning and length normalisation matter most. Stemming mainly makes the model "
    "smaller (31% fewer features at the same quality). Bigrams help the similarity-based tasks but not "
    "Naive Bayes on this small training set. We kept stemmed unigrams as the default because they are fast "
    "and their results are easy to explain. Analysts can switch to bigrams in the application."])

h2("Question 2: How can duplicate or near-duplicate documents affect classification, clustering and "
   "recommendation? How did we detect or handle them?")
p("<b>Detection in our data.</b> Section 5 shows that the real corpus has no exact or near duplicates "
  f"(maximum Jaccard {f3(dups['top_pairs'][0]['jaccard'])}). To measure the effect of duplicates anyway, "
  "<font face='Courier'>src/experiments.py</font> adds, in memory only, an exact copy and an edited copy "
  "(10% of the words randomly deleted) of 10 labelled documents, labelled <i>[SIMULATED]</i>. This gives "
  f"{dup['n_documents_augmented']} documents. The dataset files are not changed. Results:")
d, c, r, k = dup["detection"], dup["classification"], dup["recommendation"], dup["clustering"]
table([["Aspect", "Original corpus", "With duplicates, no handling", "With duplicates, our handling"],
       ["Detection", "-", "-", f"{d['detected_injected_pairs']}/{d['injected_pairs']} copy pairs found, "
                               f"{d['false_positive_pairs']} false positives"],
       [f"NB accuracy (mean of {c['splits']} splits)", f3(c["original_corpus"]),
        f"{f3(c['with_duplicates_naive_split'])} (on average {c['avg_leaked_test_docs_naive_split']:.1f} "
        "test documents have a copy in training)",
        f"{f3(c['with_duplicates_group_split'])} (duplicate groups kept together)"],
       ["Share of top-5 recommendation slots taken by copies", "-",
        pct(r["share_of_top5_slots_taken_by_copies_unfiltered"]),
        pct(r["share_of_top5_slots_taken_by_copies_filtered"]) + " (near-duplicates hidden)"],
       ["Category precision@5", f3(r["p_at_5_original"]), f3(r["p_at_5_with_duplicates"]) + " (inflated)", "-"],
       ["K-Means purity / RI", f"{f3(k['purity_original'])} / {f3(k['rand_index_original'])}",
        f"{f3(k['purity_with_duplicates'])} / {f3(k['rand_index_with_duplicates'])}",
        f"{f3(k['purity_after_dedup'])} / {f3(k['rand_index_after_dedup'])} (copies removed)"]],
      widths=[4.2 * cm, 2.6 * cm, 4.6 * cm, 4.6 * cm], caption="Duplicate-injection experiment")
bullets([
    f"<b>Classification:</b> with a normal random split, copies of test documents end up in the training "
    f"set. The classifier then 'recognises' them, and accuracy rises from {f3(c['original_corpus'])} to "
    f"{f3(c['with_duplicates_naive_split'])}. This is data leakage: the higher score is not real. Our "
    f"group-aware split removes the effect ({f3(c['with_duplicates_group_split'])}). Duplicates also give "
    "the duplicated documents more weight in the class priors and likelihoods.",
    f"<b>Clustering:</b> {k['copies_in_same_cluster_as_source']} of {k['copies_total']} copies landed in the "
    "same cluster as their source. The duplicated documents form heavy groups that pull centroids towards "
    f"them, and purity dropped from {f3(k['purity_original'])} to {f3(k['purity_with_duplicates'])}. Removing "
    "the detected copies restores the result.",
    f"<b>Recommendation:</b> without filtering, {pct(r['share_of_top5_slots_taken_by_copies_unfiltered'])} of "
    "the top-5 slots of the affected documents were taken by their own copies (two of five slots), which "
    "is useless to an analyst. The duplicates also artificially raise the precision@5 metric. Hiding results "
    "with cosine &#8805; 0.9 removed all of them.",
    f"<b>Detection quality:</b> all injected pairs were found with no false positives. The lowest Jaccard "
    f"of an edited copy was {f3(d['min_jaccard_near_copy'])}, just above the 0.5 threshold: deleting 10% of "
    "words already breaks many 5-word shingles. That is why we also use the TF-IDF cosine. A heavily edited "
    "copy could be missed by Jaccard but is still caught by cosine."])

h2("Question 3: Explain the recommendation approach. Compare content-based and collaborative "
   "recommendation.")
p("Our application uses <b>content-based</b> recommendation. Each document is a normalised TF-IDF vector, "
  "and the Top-K documents with the highest cosine similarity to the selected document (or to a typed "
  "query) are recommended, excluding the document itself and its near-duplicates. The shared terms "
  "explain every recommendation. On our corpus, "
  f"{pct(rec['p_at_5'])} of the top-5 recommendations share the category of the source document.")
table([["", "Content-based (used)", "Collaborative filtering"],
       ["Idea", "Recommend items whose content is similar to the item the user is looking at",
        "Recommend items that similar users liked / used (user-user or item-item co-occurrence)"],
       ["Data needed", "Only the item content (text) and its features",
        "A user-item interaction matrix: views, clicks, ratings, bookmarks, ideally many per user and item"],
       ["New items", "Works immediately (no cold start for items)", "Cold start: a new item has no interactions"],
       ["Strengths", "Explainable (shared terms), private, works for small user bases",
        "Finds related items with different vocabulary; captures quality and popularity; more serendipity"],
       ["Weaknesses", "Only lexical similarity; ambiguous words; 'more of the same'; long general documents "
                      "match many others", "Needs a lot of interaction data; popularity bias; cold start for "
                                           "new users"],
       ["Preferable when", "Content is rich and interaction data is missing or sparse (our case)",
        "There are many active users and logged interactions, e.g. a large news or e-commerce site"]],
      widths=[2.6 * cm, 6.7 * cm, 6.7 * cm], caption="Content-based vs collaborative recommendation")
p("Our dataset contains no user interaction data, so collaborative filtering would need invented data. "
  "Content-based filtering is therefore the honest and appropriate choice. Its limitations are visible in "
  "our results: the ambiguity of 'agent' (Section 13.3), weak matches on generic words (<i>Data center</i> "
  "for <i>Stock market</i>), and no notion of which articles consultants actually found useful. Once the "
  "knowledge library logs which articles are opened together, a <b>hybrid</b> recommender could combine "
  "both: content scores for new articles, blended with collaborative scores as usage data grows.")

h2("Question 4: How do text mining, classification, clustering and recommendation together help domain "
   "experts make better decisions?")
bullets([
    "<b>Text mining and corpus analytics</b> show what the library contains: the size of each practice area, "
    "document lengths, the most important terms, and data quality problems (reference noise, a stub "
    "article, off-domain documents, duplicates). Knowledge managers can decide what to clean and where "
    "content is missing. For example, Computing &amp; Infrastructure has only 9 articles.",
    f"<b>Classification</b> can tag new articles automatically. With about {pct(cv['accuracy'])} cross-validated "
    "accuracy it is good enough to <i>suggest</i> a practice area, which a librarian then confirms. The "
    "posterior probabilities shown in the application identify uncertain cases for manual review. This saves "
    "manual effort and keeps tagging consistent.",
    "<b>Clustering</b> shows how the content actually groups, independent of our labels. Our clusters show "
    "that sequence-model and language articles overlap strongly with ML, and that web, services and security "
    "form a theme across categories. This can lead to a new practice area (e.g. 'Digital Platforms &amp; "
    "Security') or to cross-references between areas.",
    "<b>Recommendation</b> helps consultants prepare faster. Starting from one relevant article (or a client "
    "question typed as free text) they get related reading with an explanation, without knowing the "
    "category structure.",
    "Together, these tools turn an unorganised folder of documents into a searchable, organised and "
    "monitored knowledge base. Experts spend less time searching and tagging and more time on analysis, and "
    "the evaluation metrics show how far each automatic result can be trusted."])

h2("Question 5: Compare Naive Bayes and K-Means. Key findings, limitations and improvements.")
p(f"<b>Results.</b> Naive Bayes reached accuracy {f3(m['accuracy'])}, macro precision "
  f"{f3(m['macro_precision'])}, macro recall {f3(m['macro_recall'])} and macro F1 {f3(m['macro_f1'])} on the "
  f"test split, and {f3(cv['accuracy'])} accuracy / {f3(cv['macro_f1'])} macro F1 in 5-fold "
  f"cross-validation. K-Means with K = 5 reached purity {f3(clu['purity'])} and Rand Index "
  f"{f3(clu['rand_index'])} (5-seed mean {f3(P['D']['kmeans_purity'])} / {f3(P['D']['kmeans_rand_index'])}). "
  "Supervision makes a large difference. Naive Bayes recovers the human categories much more "
  "reliably than clustering. The Rand Index looks higher than purity mainly because most document "
  "pairs belong to different classes and are correctly separated (TN), so RI should not be read as "
  "'76% correct'.")
p("<b>Key findings.</b>")
bullets(["Both methods struggle with the same borderline documents: language-technology articles built on "
         "neural networks, and the heterogeneous Finance and Computing classes. These errors reflect real "
         "topic overlap and debatable labels rather than bugs.",
         "Noise cleaning and L2 length normalisation are essential for K-Means. Naive Bayes is much less "
         "sensitive to preprocessing.",
         "Duplicates can inflate classification accuracy through leakage (+5 percentage points in our "
         "experiment) and spoil recommendations. Group-aware splitting and duplicate filtering prevent this.",
         "K-Means results depend on initialisation and K. Lowest SSE was not the best match to the labels."])
p("<b>Limitations.</b>")
bullets(["The corpus is small (71 labelled documents, 21 in the test split), so one document changes "
         "accuracy by about 4.8 points, and the classes are imbalanced (22 vs 9 documents).",
         "Labels were assigned by us and some are debatable. There is only one label per document, although "
         "many articles span two areas.",
         "Wikipedia articles are long, encyclopaedic texts. Real consulting documents (reports, emails) are "
         "shorter and noisier in other ways.",
         "Bag-of-words TF-IDF ignores word order and meaning (synonyms, ambiguity such as 'agent').",
         "K-Means assumes roughly spherical, similarly sized clusters and needs K in advance. PCA shows only "
         f"{sum(clu['explained_variance_2d']) * 100:.0f}% of the variance in 2D, so the plot overlaps more "
         "than the real clusters do.",
         "The recommendation evaluation (category precision@K) is only a proxy. We have no user judgements."])
p("<b>Suggested improvements for real-world deployment.</b>")
bullets(["Collect more documents and labels, allow multi-label tagging, and use active learning: the "
         "librarian confirms the predictions with the lowest posterior probability first.",
         "Use feature selection (chi-square or mutual information) or Complement Naive Bayes for the "
         "imbalanced classes. Compare with a linear SVM or logistic regression.",
         "For clustering, choose K with the silhouette score, use spherical (cosine) K-Means or LSA/embeddings "
         "to reduce sparsity, and label clusters automatically with their top terms.",
         "Use sentence embeddings or LSA to capture synonyms in recommendations, add MMR re-ranking for "
         "diversity, and log user clicks to move towards a hybrid recommender.",
         "Run duplicate detection (MinHash/LSH for scale) on every new upload before indexing. Store documents "
         "and the index in a database with incremental index updates instead of recomputing in memory. Monitor "
         "accuracy over time as new topics appear, and re-train regularly."])

# ================================================================ 17-19
h1("Limitations")
p("Besides the limitations discussed in Question 5: the application keeps the corpus and all models in "
  "memory and recomputes them when settings change, which is fine for about a hundred documents but would "
  "need incremental indexing for thousands. Uploaded documents are kept only for the browser session. "
  "Lemmatization uses WordNet without part-of-speech tagging (nouns, then verbs), so some forms are not "
  "reduced. The dataset is about 3.9 MB of text, below the optional 5 MB guideline, although the document "
  "count requirement is met.")
h1("Future Improvements")
bullets(["Persistent storage of uploads and labels with an annotation screen for librarians.",
         "Multi-label classification and hierarchical categories.",
         "Semantic features (word or sentence embeddings) next to TF-IDF, and a hybrid recommender once usage "
         "data exists.",
         "Automatic cluster naming and trend monitoring over time (new topics appearing in the library).",
         "Scalable near-duplicate detection with MinHash and locality-sensitive hashing."])
h1("Conclusion")
p("We built a complete, working Streamlit text mining system for a technology knowledge library. The "
  "system supports ingestion with duplicate detection, configurable preprocessing, manually implemented "
  "TF-IDF with an inverted index and cosine similarity, Naive Bayes classification "
  f"(accuracy {f3(m['accuracy'])}, CV {f3(cv['accuracy'])}), K-Means clustering (purity {f3(clu['purity'])}, "
  f"Rand Index {f3(clu['rand_index'])}) and content-based recommendation (category precision@5 "
  f"{f3(rec['p_at_5'])}). The experiments show that noise cleaning and length normalisation matter more than "
  "the choice between stemming and lemmatization, that supervised classification matches the human "
  "categories much better than clustering, and that duplicates must be handled to avoid misleading "
  "results. The whole workflow can be run and explored from the Streamlit front end.")

# ================================================================ 20 references
h1("References")
for ref in [
    "C. D. Manning, P. Raghavan and H. Schuetze, <i>Introduction to Information Retrieval</i>, Cambridge "
    "University Press, 2008 (Ch. 6 TF-IDF and vector space model, Ch. 13 Naive Bayes, Ch. 16 flat "
    "clustering, Purity and Rand Index).",
    "M. F. Porter, 'An algorithm for suffix stripping', <i>Program</i>, 14(3), pp. 130-137, 1980.",
    "D. Arthur and S. Vassilvitskii, 'k-means++: The advantages of careful seeding', <i>Proc. ACM-SIAM SODA</i>, "
    "2007.",
    "A. Z. Broder, 'On the resemblance and containment of documents', <i>Proc. Compression and Complexity of "
    "Sequences</i>, 1997.",
    "W. M. Rand, 'Objective criteria for the evaluation of clustering methods', <i>Journal of the American "
    "Statistical Association</i>, 66(336), pp. 846-850, 1971.",
    "S. Bird, E. Klein and E. Loper, <i>Natural Language Processing with Python</i> (NLTK), O'Reilly, 2009.",
    "F. Ricci, L. Rokach and B. Shapira (eds.), <i>Recommender Systems Handbook</i>, Springer, 2015.",
    "Wikipedia contributors, English Wikipedia articles listed in data/processed/corpus_metadata.csv (exact "
    "revision URLs included), licensed under CC BY-SA 4.0.",
    "Streamlit documentation, https://docs.streamlit.io",
]:
    p(ref, SMALL)

# ================================================================ appendix
story.append(PageBreak())
p("<b>Appendix A - BITS Virtual Lab execution evidence</b>", H1)
lab_images = sorted([*EVIDENCE.glob("*.png"), *EVIDENCE.glob("*.jpg"), *EVIDENCE.glob("*.jpeg")]) \
    if EVIDENCE.exists() else []
if lab_images:
    p("The following screenshots were taken while running the application in the BITS Virtual Lab.")
    for img in lab_images:
        figure(img, img.stem.replace("_", " "), width=15 * cm, max_height=11 * cm)
else:
    p("<font color='#b00000'><b>To be completed by the group:</b> Virtual Lab screenshots are not yet "
      "included. Put the screenshots in <font face='Courier'>evidence/virtual_lab/</font> and run "
      "<font face='Courier'>python scripts/build_report.py</font> again. They will be inserted here "
      "automatically.</font>")
p("The screen recording is submitted as a separate file in the <font face='Courier'>evidence/</font> folder "
  "of the ZIP. The screenshots in the main part of this report were taken from a local run of the same "
  "code and are labelled 'local run'.", SMALL)
p("<b>Appendix B - How to run</b>", H1)
formula("python -m venv .venv\n.venv\\Scripts\\activate        (Linux: source .venv/bin/activate)\n"
        "pip install -r requirements.txt\nstreamlit run app.py\n\n"
        "python -m pytest -q                 # 40 unit tests\n"
        "python scripts/run_experiments.py   # results + figures\n"
        "python scripts/build_report.py      # this PDF")


def page_number(canvas, doc):
    if doc.page > 1:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(2 * cm, 1.2 * cm, "IR Assignment 2 - Group G25")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()


doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                        topMargin=2 * cm, bottomMargin=2 * cm,
                        title="IR Assignment 2 - Group G25 - Technical Report", author="Group G25")
doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
print("Report written to", OUTPUT)
