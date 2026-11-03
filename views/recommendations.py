import streamlit as st

from config import NEAR_DUPLICATE_COSINE
from src import visualization as viz
from src.app_state import doc_label, get_model, preprocess_query
from src.recommendation import category_precision_at_k, recommend, recommend_for_query

st.title("Document Recommendations")
model = get_model()
docs = model.docs
vectors = model.tfidf.vectors

doc_tab, text_tab, about_tab = st.tabs(["Related documents", "Search by text", "How it works"])

with doc_tab:
    c1, c2 = st.columns([3, 1])
    i = c1.selectbox("Select a document", range(len(docs)), format_func=lambda i: doc_label(docs, i))
    k = c2.slider("Top-K", 1, 15, 5)
    c1, c2 = st.columns([1, 3])
    hide_dups = c1.checkbox("Hide near-duplicates", True)
    threshold = c2.slider("Duplicate threshold (cosine)", 0.5, 1.0, NEAR_DUPLICATE_COSINE, 0.01,
                          disabled=not hide_dups)

    st.markdown(f"**Selected:** {docs['doc_id'].iloc[i]} - *{docs['title'].iloc[i]}* "
                f"({docs['category'].iloc[i]}, {docs['word_count'].iloc[i]:,} words)")
    top_terms = [t for t, _ in model.tfidf.top_terms(i, 10)]
    st.caption("Top TF-IDF terms of the selected document: " + ", ".join(top_terms))

    recs, skipped = recommend(i, model.similarity, docs, vectors, k, hide_dups, threshold)
    if skipped:
        st.info(f"Hidden as near-duplicates of the selected document: {', '.join(skipped)}")
    if recs.empty:
        st.warning("No recommendations available (the corpus has too few documents).")
    else:
        same = recs["same_category"].mean()
        st.write(f"{recs['same_category'].sum()} of {len(recs)} recommendations "
                 f"({same:.0%}) are in the same category as the selected document.")
        st.dataframe(recs.drop(columns=["index"]).round(4), hide_index=True, width="stretch",
                     column_config={"similarity": st.column_config.ProgressColumn(
                         "similarity", min_value=0.0, max_value=1.0, format="%.4f")})
        st.plotly_chart(viz.recommendation_scores(recs))

        st.markdown("**Why were these recommended?** The score is the cosine similarity of the two "
                    "TF-IDF vectors, i.e. the sum over shared terms of the product of their weights. "
                    "`shared_terms` lists the terms contributing most to that sum.")
        read = st.selectbox("Read a recommended document", recs["index"].tolist(),
                            format_func=lambda j: doc_label(docs, j))
        with st.expander(f"Text of {docs['title'].iloc[read]}", expanded=False):
            st.write(docs["text"].iloc[read][:6000] + (" ..." if len(docs["text"].iloc[read]) > 6000 else ""))

    st.divider()
    p5 = category_precision_at_k(model.similarity, docs["category"], 5, model.eval_mask)
    st.metric("Category precision@5 over all labelled documents", f"{p5:.3f}",
              help="Average share of the top-5 recommendations that have the same category as the "
                   "selected document. Used as a simple offline quality check.")

with text_tab:
    st.markdown("Enter any text (for example an abstract or a client question). It is preprocessed "
                "with the current settings, weighted with the corpus idf, and scored against the "
                "documents through the **inverted index** (only postings of the query terms are visited).")
    query = st.text_area("Query text", "How do recurrent neural networks handle long sequences "
                                       "in speech recognition?")
    k2 = st.slider("Top-K results", 1, 15, 5, key="k_query")
    if query.strip():
        q_vec = model.tfidf.transform(preprocess_query(query))
        if not q_vec:
            st.warning("None of the query terms are in the index vocabulary.")
        else:
            st.caption("Query terms used: " + ", ".join(
                f"{t} ({w:.3f})" for t, w in sorted(q_vec.items(), key=lambda x: -x[1])[:15]))
            results = recommend_for_query(q_vec, model.index, docs, vectors, k2)
            st.dataframe(results.drop(columns=["index"]).round(4), hide_index=True,
                         width="stretch")

with about_tab:
    st.markdown("""
### Content-based recommendation (used here)
Each document is represented by its manually computed, L2-normalised TF-IDF vector.
For a selected document *d*, every other document *d'* gets the score cos(d, d') = Σ_t w(t,d)·w(t,d'),
and the K highest-scoring documents are returned. The document itself is always excluded, and
near-duplicates (cosine above the threshold) can be hidden so that the list is not filled with copies.

**Strengths for this dataset:** no user data is needed, the system works for a brand-new document
straight away, and every recommendation can be explained through the shared terms.

**Limitations:**
- It only matches words, not meaning: *"automobile"* and *"car"* are unrelated unless both appear.
  Long general articles (e.g. *Machine learning*) share vocabulary with many others and appear often.
- Ambiguous words cause mistakes. *Agent (economics)* and *Intelligent agent* share the word "agent"
  but belong to different domains.
- It cannot learn what analysts actually found useful, and it tends to recommend "more of the same"
  (low diversity / serendipity).
- Scores depend on the preprocessing settings (stemming, stop-words, n-grams) and on min df.

### Collaborative filtering (not used)
Collaborative filtering recommends documents that *similar users* read or rated. It needs a
user-document interaction matrix (views, clicks, ratings, bookmarks). It can find related items
whose text looks different and can capture quality and popularity. However, it suffers from the
*cold-start* problem for new documents and new users and needs many interactions.

Our corpus has **no user interaction data**, so a collaborative approach would have to use invented
data. Content-based filtering is therefore the appropriate choice. In a real deployment, once
the knowledge library logs which articles consultants open, a **hybrid** approach could be used:
content-based scores for new documents, blended with collaborative scores as interactions build up.
""")
