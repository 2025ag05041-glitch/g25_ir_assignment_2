"""Session state and caching helpers for the Streamlit pages.

The corpus = base dataset + documents uploaded in this session. Every
expensive result is cached on a key made from the corpus fingerprint and the
settings, so changing the corpus or any setting automatically gives fresh
results while unchanged settings are reused.
"""
import hashlib

import streamlit as st

from config import NEAR_DUPLICATE_COSINE, NEAR_DUPLICATE_JACCARD
from src.duplicates import classify_pairs, duplicate_groups, pairwise_similarity
from src.ingestion import load_base_corpus, merge_uploads
from src.pipeline import CorpusModel, FeatureConfig, run_classification, run_clustering
from src.preprocessing import PreprocessConfig, preprocess, raw_tokens, vocabulary_stats


@st.cache_data(show_spinner="Loading dataset...")
def base_corpus():
    return load_base_corpus()


def get_corpus():
    return merge_uploads(base_corpus(), st.session_state.get("uploaded_records", []))


def fingerprint(docs):
    h = hashlib.md5()
    for doc_id, cat, text in zip(docs["doc_id"], docs["category"], docs["text"]):
        h.update(f"{doc_id}|{cat}|{len(text)}|".encode())
        h.update(hashlib.md5(text.encode("utf-8")).digest())
    return h.hexdigest()


def get_prep_config():
    return st.session_state.get("prep_config", PreprocessConfig())


def get_feature_config():
    return st.session_state.get("feature_config", FeatureConfig())


@st.cache_resource(max_entries=4, show_spinner="Preprocessing corpus and building TF-IDF / index...")
def _build_model(key, prep, feat, _docs):
    return CorpusModel(_docs, prep, feat)


def get_model():
    docs = get_corpus()
    return _build_model(fingerprint(docs), get_prep_config(), get_feature_config(), docs)


@st.cache_data(max_entries=4, show_spinner="Computing raw vocabulary...")
def _raw_stats(key, _docs):
    return vocabulary_stats([raw_tokens(t) for t in _docs["text"]])


def get_raw_stats():
    docs = get_corpus()
    return _raw_stats(fingerprint(docs), docs)


def get_processed_stats():
    return vocabulary_stats(get_model().docs_terms)


# --------------------------------------------------- text mining settings

NB_DEFAULTS = {"test_size": 0.3, "seed": 42, "alpha": 1.0, "group_duplicates": True}
KM_DEFAULTS = {"k": 5, "init": "k-means++", "n_init": 10, "max_iter": 100, "seed": 42}


def get_nb_params():
    return {**NB_DEFAULTS, **st.session_state.get("nb_params", {})}


def get_km_params():
    return {**KM_DEFAULTS, **st.session_state.get("km_params", {})}


@st.cache_resource(max_entries=8, show_spinner="Detecting duplicates...")
def _duplicates(key, jac, cos, _model):
    pairs = pairwise_similarity(_model.docs, _model.similarity)
    return classify_pairs(pairs, jac, cos)


def get_duplicate_pairs(jaccard=NEAR_DUPLICATE_JACCARD, cosine=NEAR_DUPLICATE_COSINE):
    model = get_model()
    key = (fingerprint(model.docs), model.prep_config, model.feature_config)
    return _duplicates(key, jaccard, cosine, model)


def get_duplicate_groups():
    pairs = get_duplicate_pairs()
    flagged = pairs[pairs["type"] != "Distinct"]
    return duplicate_groups(len(get_model().docs),
                            [(int(i), int(j)) for i, j in zip(flagged["i"], flagged["j"])])


@st.cache_resource(max_entries=8, show_spinner="Training Naive Bayes...")
def _classification(key, params, _model, _groups):
    p = dict(params)
    return run_classification(_model, p["test_size"], p["seed"], p["alpha"],
                              _groups if p["group_duplicates"] else None)


def get_classification():
    model = get_model()
    params = get_nb_params()
    key = (fingerprint(model.docs), model.prep_config, model.feature_config)
    groups = get_duplicate_groups()
    return _classification((key, tuple(groups)), tuple(sorted(params.items())), model, groups)


@st.cache_resource(max_entries=8, show_spinner="Running K-Means...")
def _clustering(key, params, _model):
    p = dict(params)
    return run_clustering(_model, p["k"], p["init"], p["n_init"], p["max_iter"], p["seed"])


def get_clustering():
    model = get_model()
    params = get_km_params()
    key = (fingerprint(model.docs), model.prep_config, model.feature_config)
    return _clustering(key, tuple(sorted(params.items())), model)


def preprocess_query(text):
    return preprocess(text, get_prep_config())


def sidebar_status():
    docs = get_corpus()
    uploaded = len(st.session_state.get("uploaded_records", []))
    st.sidebar.markdown("**Current settings**")
    st.sidebar.caption(f"Corpus: {len(docs)} documents"
                       + (f" ({uploaded} uploaded)" if uploaded else ""))
    st.sidebar.caption("Preprocessing: " + get_prep_config().describe())
    feat = get_feature_config()
    st.sidebar.caption(f"TF-IDF: {feat.tf_scheme} tf, min df {feat.min_df}, "
                       + ("L2 normalised" if feat.normalize else "not normalised"))


def doc_label(docs, i):
    return f"{docs['doc_id'].iloc[i]} - {docs['title'].iloc[i]}"
