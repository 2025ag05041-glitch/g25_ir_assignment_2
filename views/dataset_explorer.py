import streamlit as st

from config import CATEGORIES, NEAR_DUPLICATE_COSINE, NEAR_DUPLICATE_JACCARD
from src.app_state import base_corpus, get_corpus, get_duplicate_pairs, get_model
from src.ingestion import METADATA_COLUMNS, corpus_statistics, parse_upload

st.title("Dataset & Upload")

docs = get_corpus()
browse_tab, upload_tab, dup_tab = st.tabs(["Browse documents", "Upload documents",
                                           "Duplicate analysis"])

with browse_tab:
    st.markdown("Each document has a unique ID, a title, a category label and source metadata. "
                "The text is stored separately in the `Dataset/` folder; metadata is saved in "
                "`data/processed/corpus_metadata.csv`.")
    col1, col2 = st.columns([1, 2])
    categories = col1.multiselect("Filter by category", sorted(docs["category"].unique()))
    query = col2.text_input("Search in title or text", placeholder="e.g. neural network")
    view = docs
    if categories:
        view = view[view["category"].isin(categories)]
    if query.strip():
        q = query.strip().lower()
        view = view[view["title"].str.lower().str.contains(q, regex=False)
                    | view["text"].str.lower().str.contains(q, regex=False)]
    st.write(f"{len(view)} of {len(docs)} documents")
    st.dataframe(view[METADATA_COLUMNS], hide_index=True, width="stretch",
                 column_config={"source_url": st.column_config.LinkColumn("source_url")})

    if len(view):
        choice = st.selectbox("Open a document", view.index,
                              format_func=lambda i: f"{docs.at[i, 'doc_id']} - {docs.at[i, 'title']}")
        doc = docs.loc[choice]
        st.markdown(f"**{doc['title']}** &nbsp;|&nbsp; {doc['category']} &nbsp;|&nbsp; "
                    f"{doc['word_count']:,} words &nbsp;|&nbsp; source: {doc['source']}")
        st.text_area("Document text", doc["text"], height=320)

    st.download_button("Download metadata (CSV)", docs[METADATA_COLUMNS].to_csv(index=False),
                       "corpus_metadata.csv", "text/csv")

with upload_tab:
    st.markdown("""
Upload extra documents to add them to the corpus for this session. Supported formats:
- **.txt** - one document per file (first line is used as the title)
- **.csv** - one row per document with a `text` column and optional `title`, `category`, `doc_id`, `source`
- **.json** - a list of objects with the same fields

Uploaded documents get IDs `UPL001`, `UPL002`, ... unless a unique `doc_id` is given.
All pages (preprocessing, mining, recommendations) use the updated corpus.
""")
    files = st.file_uploader("Choose files", type=["txt", "csv", "json"], accept_multiple_files=True)
    default_category = st.selectbox("Category for uploaded .txt files (or rows without a category)",
                                    ["Unlabelled"] + CATEGORIES)
    if st.button("Add to corpus", disabled=not files):
        added, messages = [], []
        for f in files:
            records, warnings = parse_upload(f.name, f.getvalue(), default_category)
            added += records
            messages += warnings
        st.session_state["uploaded_records"] = st.session_state.get("uploaded_records", []) + added
        # shown after the rerun below
        st.session_state["upload_messages"] = messages + [f"Added {len(added)} document(s)."]
        st.rerun()

    for m in st.session_state.pop("upload_messages", []):
        (st.success if m.startswith("Added") else st.warning)(m)

    uploaded = st.session_state.get("uploaded_records", [])
    if uploaded:
        st.write(f"{len(uploaded)} uploaded document(s) in this session:")
        new_docs = docs[~docs["doc_id"].isin(base_corpus()["doc_id"])]
        st.dataframe(new_docs[METADATA_COLUMNS], hide_index=True, width="stretch")
        st.caption("Unlabelled uploads are used for browsing and recommendation only.")
        if st.button("Remove uploaded documents"):
            st.session_state["uploaded_records"] = []
            st.rerun()

    st.subheader("Corpus statistics")
    stats = corpus_statistics(docs)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", stats["documents"])
    c2.metric("Shortest document", f"{stats['min_words']} words")
    c3.metric("Longest document", f"{stats['max_words']:,} words")
    c4.metric("Empty documents", stats["empty_documents"])

with dup_tab:
    st.markdown(f"""
**How duplicates are detected**
- *Exact duplicates*: identical MD5 hash of the text after lowercasing and removing punctuation/extra spaces.
- *Near duplicates*: Jaccard similarity of word 5-gram shingle sets, J(A,B) = |A ∩ B| / |A ∪ B|,
  and cosine similarity of the TF-IDF vectors. A pair is flagged when Jaccard ≥ threshold **or** cosine ≥ threshold.

**How they are handled**
- Flagged documents are put in the same *duplicate group*; the train/test split keeps a group on one side,
  so a copy of a test document can never be in the training set (no leakage).
- The recommender can hide documents whose cosine similarity is above the duplicate threshold.
- The original files are never modified.
""")
    c1, c2 = st.columns(2)
    jac = c1.slider("Jaccard threshold (5-word shingles)", 0.1, 1.0, NEAR_DUPLICATE_JACCARD, 0.05)
    cos = c2.slider("Cosine threshold (TF-IDF)", 0.5, 1.0, NEAR_DUPLICATE_COSINE, 0.01)
    get_model()
    pairs = get_duplicate_pairs(jac, cos)
    flagged = pairs[pairs["type"] != "Distinct"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Document pairs compared", f"{len(pairs):,}")
    c2.metric("Exact duplicate pairs", int((pairs["type"] == "Exact duplicate").sum()))
    c3.metric("Near-duplicate pairs", int((pairs["type"] == "Near duplicate").sum()))
    show_cols = ["doc_a", "title_a", "doc_b", "title_b", "jaccard", "cosine", "type"]
    if len(flagged):
        st.dataframe(flagged[show_cols].round(3), hide_index=True, width="stretch")
    else:
        st.info("No exact or near-duplicate pairs at these thresholds.")
    st.markdown("**Most similar document pairs** (useful to see how close the closest documents are)")
    st.dataframe(pairs.sort_values("jaccard", ascending=False).head(10)[show_cols].round(3),
                 hide_index=True, width="stretch")
    st.caption("Tip: upload one of the files from the Dataset folder in the Upload tab to see an "
               "exact duplicate being detected. The Performance Analysis page has an experiment that "
               "measures the effect of duplicates on all three tasks.")
