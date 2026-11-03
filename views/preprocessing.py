import numpy as np
import pandas as pd
import streamlit as st

from src import visualization as viz
from src.app_state import (doc_label, get_feature_config, get_model, get_prep_config,
                           get_processed_stats, get_raw_stats)
from src.features import cosine_similarity
from src.pipeline import FeatureConfig
from src.preprocessing import (PreprocessConfig, lemmatizer_name, preprocess, preprocess_steps,
                               stopwords_source)

st.title("Preprocessing & Feature Engineering")
st.caption(f"Stop-words: {stopwords_source()}. Lemmatizer: {lemmatizer_name()}.")

prep = get_prep_config()
feat = get_feature_config()

with st.form("settings"):
    st.markdown("**Preprocessing settings** (applied to the whole application)")
    c1, c2, c3 = st.columns(3)
    lowercase = c1.checkbox("Lowercasing", prep.lowercase)
    remove_noise = c1.checkbox("Remove URLs and citation markers like [12]", prep.remove_noise)
    strip_refs = c1.checkbox("Strip References / External links sections", prep.strip_references)
    tokenizer = c2.radio("Tokenization", ["regex", "whitespace"],
                         index=["regex", "whitespace"].index(prep.tokenizer),
                         format_func={"regex": "Regex (words only)",
                                      "whitespace": "Whitespace split (keeps punctuation)"}.get)
    remove_stop = c2.checkbox("Stop-word removal", prep.remove_stopwords)
    remove_numbers = c2.checkbox("Remove tokens containing digits", prep.remove_numbers)
    min_len = c2.number_input("Minimum token length", 1, 5, prep.min_token_length)
    normalizer = c3.radio("Word normalisation", ["none", "stem", "lemma"],
                          index=["none", "stem", "lemma"].index(prep.normalizer),
                          format_func={"none": "None", "stem": "Stemming (Porter)",
                                       "lemma": f"Lemmatization ({lemmatizer_name()})"}.get)
    ngram = c3.selectbox("N-grams", [1, 2, 3], index=prep.ngram_max - 1,
                         format_func={1: "Unigrams", 2: "Unigrams + bigrams",
                                      3: "Unigrams + bigrams + trigrams"}.get)
    st.markdown("**TF-IDF settings**")
    c1, c2, c3 = st.columns(3)
    tf_scheme = c1.radio("TF weighting", ["log", "raw"], index=["log", "raw"].index(feat.tf_scheme),
                         format_func={"log": "1 + log10(tf)", "raw": "raw count tf"}.get)
    min_df = c2.number_input("Minimum document frequency (min df)", 1, 10, feat.min_df)
    normalize = c3.checkbox("L2 document-length normalisation", feat.normalize)
    if st.form_submit_button("Apply settings", type="primary"):
        st.session_state["prep_config"] = PreprocessConfig(
            lowercase, tokenizer, remove_noise, strip_refs, remove_stop, remove_numbers,
            int(min_len), normalizer, int(ngram))
        st.session_state["feature_config"] = FeatureConfig(int(min_df), 1.0, tf_scheme, normalize)
        st.rerun()

if st.button("Reset to default settings"):
    st.session_state.pop("prep_config", None)
    st.session_state.pop("feature_config", None)
    st.rerun()

model = get_model()
docs = model.docs
raw = get_raw_stats()
processed = get_processed_stats()

st.subheader("Before vs after preprocessing")
st.caption("'Before' = plain whitespace split of the raw text. 'After' = output of the current pipeline "
           "(including n-grams if selected).")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Vocabulary size", f"{processed['vocabulary_size']:,}",
          f"{processed['vocabulary_size'] - raw['vocabulary_size']:,} vs raw ({raw['vocabulary_size']:,})")
c2.metric("Total tokens", f"{processed['total_tokens']:,}",
          f"{processed['total_tokens'] - raw['total_tokens']:,} vs raw ({raw['total_tokens']:,})")
c3.metric("Avg. tokens / document", f"{processed['avg_doc_length']:,.0f}",
          f"{processed['avg_doc_length'] - raw['avg_doc_length']:,.0f} vs raw")
c4.metric("TF-IDF features (df ≥ min df)", f"{len(model.tfidf.vocabulary):,}")

compare = pd.DataFrame({
    "statistic": ["Vocabulary size", "Total tokens", "Average tokens per document",
                  "Terms occurring once (hapax)"],
    "before": [raw["vocabulary_size"], raw["total_tokens"], round(raw["avg_doc_length"]),
               raw["hapax_count"]],
    "after": [processed["vocabulary_size"], processed["total_tokens"],
              round(processed["avg_doc_length"]), processed["hapax_count"]],
})
compare["change %"] = ((compare["after"] - compare["before"]) / compare["before"] * 100).round(1)
st.dataframe(compare, hide_index=True, width="stretch")

left, right = st.columns(2)
left.plotly_chart(viz.top_terms_bar(raw["counts"], 20, "Top 20 terms before preprocessing"))
right.plotly_chart(viz.top_terms_bar(processed["counts"], 20, "Top 20 terms after preprocessing",
                                     color="#eb6834"))
st.plotly_chart(viz.rank_frequency(raw["counts"], processed["counts"]))

st.subheader("Step-by-step view of one document")
i = st.selectbox("Document", range(len(docs)), format_func=lambda i: doc_label(docs, i))
steps = preprocess_steps(docs["text"].iloc[i], model.prep_config)
step_table = pd.DataFrame({
    "step": ["Raw text (whitespace tokens)", "Tokenization", "After stop-word / number / length filter",
             "After stemming / lemmatization", "Final terms (with n-grams)"],
    "tokens": [len(docs["text"].iloc[i].split()), len(steps["tokens"]), len(steps["after_filtering"]),
               len(steps["after_normalization"]), len(steps["terms"])],
    "unique": [len(set(docs["text"].iloc[i].split())), len(set(steps["tokens"])),
               len(set(steps["after_filtering"])), len(set(steps["after_normalization"])),
               len(set(steps["terms"]))],
})
st.dataframe(step_table, hide_index=True, width="stretch")
c1, c2 = st.columns(2)
c1.text_area("Original text (first 1500 characters)", docs["text"].iloc[i][:1500], height=260)
c2.text_area("Processed terms (first 150)", " ".join(steps["terms"][:150]), height=260)

st.subheader("Manual TF-IDF")
st.markdown("Implemented from scratch in `src/features.py` (no library TF-IDF):")
st.latex(r"w_{tf}(t,d) = 1 + \log_{10} tf(t,d) \;\; (\text{or raw } tf) \qquad "
         r"idf(t) = \log_{10}\frac{N}{df(t)}")
st.latex(r"w(t,d) = w_{tf}(t,d)\cdot idf(t) \qquad "
         r"\hat w(t,d) = \frac{w(t,d)}{\sqrt{\sum_{t'} w(t',d)^2}} \qquad "
         r"\cos(d_1,d_2) = \sum_t \hat w(t,d_1)\,\hat w(t,d_2)")
st.caption(f"N = {model.tfidf.n_docs} documents. A term that occurs in every document gets idf = 0 "
           "and therefore no weight.")
tfidf = model.tfidf
rows = []
for term, weight in tfidf.top_terms(i, 15):
    rows.append({"term": term, "tf": tfidf.tf_list[i][term], "df": tfidf.df[term],
                 "idf": round(tfidf.idf[term], 4), "normalised tf-idf": round(weight, 4)})
st.markdown(f"Top TF-IDF terms of **{docs['title'].iloc[i]}**")
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.plotly_chart(viz.df_histogram([tfidf.df[t] for t in tfidf.terms]))

st.subheader("Inverted index")
istats = model.index.stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Indexed terms", f"{istats['terms']:,}")
c2.metric("Total postings", f"{istats['postings']:,}")
c3.metric("Avg. postings list length", f"{istats['avg_postings_length']:.1f}")
c4.metric("Longest postings list", istats["max_postings_length"])
term = st.text_input("Look up a term (after preprocessing, e.g. 'neural' -> 'neural' / stem 'network')",
                     "network")
lookup = preprocess(term, model.prep_config)
key = lookup[0] if lookup else term.lower()
postings = model.index.lookup(key)
if postings:
    st.write(f"Index term **{key}** appears in {len(postings)} documents (df = {len(postings)}, "
             f"idf = {tfidf.idf[key]:.4f}).")
    st.dataframe(pd.DataFrame([{"doc_id": docs["doc_id"].iloc[d], "title": docs["title"].iloc[d],
                                "tf": tf, "tf-idf weight": round(w, 4)} for d, tf, w in postings])
                 .sort_values("tf-idf weight", ascending=False), hide_index=True,
                 width="stretch")
else:
    st.info(f"'{key}' is not in the index (it may be a stop-word, too rare for min df, or occur in "
            "every document so that idf = 0).")

st.subheader("Cosine similarity")
c1, c2 = st.columns(2)
a = c1.selectbox("Document A", range(len(docs)), format_func=lambda i: doc_label(docs, i), key="cos_a")
b = c2.selectbox("Document B", range(len(docs)), index=min(1, len(docs) - 1),
                 format_func=lambda i: doc_label(docs, i), key="cos_b")
score = cosine_similarity(tfidf.vectors[a], tfidf.vectors[b])
st.metric("Cosine similarity", f"{score:.4f}")
order = np.argsort(docs["category"].values, kind="stable")
st.plotly_chart(viz.similarity_heatmap(model.similarity[np.ix_(order, order)],
                                       [doc_label(docs, j) for j in order]))
st.caption("Bright blocks on the diagonal mean documents of the same category are more similar "
           "to each other than to the rest of the corpus.")
