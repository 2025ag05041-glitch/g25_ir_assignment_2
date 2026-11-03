import pandas as pd
import streamlit as st

from config import ROOT_DIR
from src import visualization as viz
from src.app_state import get_corpus
from src.ingestion import corpus_statistics

st.title("Text Mining & Document Intelligence System")
st.caption("Information Retrieval - Assignment 2 | Group G25")

st.markdown("""
**Business context.** Our group works as the text mining team of a technology consulting firm.
The firm keeps an internal knowledge library of reference articles on machine learning,
language technology, AI systems, computing infrastructure and finance. Consultants use it to prepare
for client projects (for example a fintech analytics engagement or an NLP proof of concept).
The library grows quickly. New articles must be tagged with the right practice area, analysts want
to see which topics occur in the collection, and when they read one article they want related reading.

This application supports that workflow:
1. **Ingest** documents (the base collection plus uploaded files) and check them for duplicates.
2. **Preprocess** the text with configurable steps and build **manual TF-IDF** vectors and an **inverted index**.
3. **Classify** documents into practice areas with a manually implemented **Naive Bayes** classifier.
4. **Cluster** documents with a manually implemented **K-Means** and evaluate with **Purity** and **Rand Index**.
5. **Recommend** related documents with content-based cosine similarity.
""")

docs = get_corpus()
stats = corpus_statistics(docs)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Documents", stats["documents"])
c2.metric("Categories", stats["categories"])
c3.metric("Total words", f"{stats['total_words']:,}")
c4.metric("Raw text size", f"{stats['total_mb']:.2f} MB")
c5.metric("Avg. words / document", f"{stats['avg_words']:,.0f}")

left, right = st.columns(2)
left.plotly_chart(viz.category_distribution(docs))
right.plotly_chart(viz.length_histogram(docs))

st.subheader("Category summary")
summary = (docs.groupby("category")
           .agg(documents=("doc_id", "count"), avg_words=("word_count", "mean"),
                min_words=("word_count", "min"), max_words=("word_count", "max"))
           .round(0).sort_values("documents", ascending=False))
st.dataframe(summary, width="stretch")
st.caption("The two 'General (Off-domain)' articles are kept for browsing and recommendation but are "
           "excluded from classification and clustering evaluation, because two documents are too few "
           "to train or test a class.")

with st.expander("Group members"):
    try:
        team = pd.read_excel(ROOT_DIR / "G025_Contribution.xlsx")
        st.dataframe(team, hide_index=True, width="stretch")
    except Exception as exc:  # file missing or openpyxl not installed
        st.write(f"Contribution file could not be read: {exc}")
