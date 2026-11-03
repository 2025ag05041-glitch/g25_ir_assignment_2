"""Tests for ingestion, preprocessing, duplicates, recommendation and the
full pipeline on the real dataset."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.duplicates import (classify_pairs, duplicate_groups, jaccard,  # noqa: E402
                            make_near_duplicate, normalized_hash, pairwise_similarity, shingles)
from src.ingestion import (add_length_columns, load_base_corpus, merge_uploads,  # noqa: E402
                           parse_upload, parse_wikipedia_text)
from src.pipeline import CorpusModel, run_classification, run_clustering  # noqa: E402
from src.preprocessing import PreprocessConfig, make_ngrams, preprocess  # noqa: E402
from src.recommendation import recommend  # noqa: E402


# ------------------------------------------------------------- ingestion
def test_parse_wikipedia_text_moves_footer_to_metadata():
    text = 'Stemming\nBody text here.\n\nRetrieved from "https://en.wikipedia.org/w/index.php?\ntitle=Stemming&oldid=123"'
    title, body, url, rev = parse_wikipedia_text(text)
    assert title == "Stemming" and "Retrieved" not in body
    assert url.endswith("oldid=123") and " " not in url and rev == "123"


def test_parse_upload_formats_and_empty_rows():
    recs, warn = parse_upload("a.txt", "Title line\nsome text".encode("utf-8-sig"), "Machine Learning")
    assert recs[0]["title"] == "Title line" and recs[0]["category"] == "Machine Learning"
    csv = b"title,text,category\nA,hello world,X\nB,,Y\n"
    recs, warn = parse_upload("d.csv", csv)
    assert len(recs) == 1 and len(warn) == 1
    recs, _ = parse_upload("d.json", json.dumps([{"text": "abc", "title": "T"}]).encode())
    assert recs[0]["title"] == "T"
    recs, warn = parse_upload("x.pdf", b"...")
    assert recs == [] and warn


def test_merge_uploads_gives_unique_ids():
    base = add_length_columns(pd.DataFrame([{"doc_id": "DOC001", "title": "t", "category": "c",
                                             "text": "x y"}]))
    merged = merge_uploads(base, [{"doc_id": "DOC001", "title": "u", "category": "c", "text": "z"},
                                  {"doc_id": "", "title": "v", "category": "c", "text": "w"}])
    assert merged["doc_id"].is_unique and len(merged) == 3


# ---------------------------------------------------------- preprocessing
def test_preprocessing_options():
    text = "The Networks were running [12] see https://example.com in 2024!"
    base = preprocess(text, PreprocessConfig(normalizer="none"))
    assert "the" not in base and "2024" not in base and "12" not in base
    assert not any("example" in t for t in base)
    assert "network" in preprocess(text, PreprocessConfig(normalizer="stem"))
    keep = preprocess(text, PreprocessConfig(remove_stopwords=False, lowercase=False,
                                             normalizer="none"))
    assert "The" in keep
    lemma = preprocess("networks studies", PreprocessConfig(normalizer="lemma"))
    assert lemma == ["network", "study"]


def test_ngrams():
    assert make_ngrams(["a", "b", "c"], 2) == ["a", "b", "c", "a_b", "b_c"]
    assert "a_b_c" in make_ngrams(["a", "b", "c"], 3)


def test_empty_text_does_not_crash():
    assert preprocess("", PreprocessConfig()) == []
    assert preprocess("the of and", PreprocessConfig()) == []


# ------------------------------------------------------------- duplicates
def test_duplicate_detection():
    text = " ".join(f"word{i}" for i in range(200))
    assert normalized_hash("Hello, World!") == normalized_hash("hello world")
    near = make_near_duplicate(text, 0.05, seed=1)
    assert jaccard(shingles(text), shingles(near)) > 0.5
    df = pd.DataFrame({"doc_id": ["A", "B", "C"], "title": ["a", "b", "c"],
                       "text": [text, text, "something completely different here"]})
    pairs = classify_pairs(pairwise_similarity(df))
    assert pairs.loc[(pairs.doc_a == "A") & (pairs.doc_b == "B"), "type"].item() == "Exact duplicate"
    assert duplicate_groups(3, [(0, 1)]) == [0, 0, 2]


# -------------------------------------------------- full pipeline (real data)
@pytest.fixture(scope="module")
def model():
    return CorpusModel(load_base_corpus())


def test_real_corpus_loaded(model):
    docs = model.docs
    assert len(docs) >= 50 and docs["doc_id"].is_unique
    assert (docs["word_count"] > 0).all()
    assert "Unlabelled" not in set(docs["category"])
    assert model.eval_mask.sum() == len(docs) - 2       # two off-domain documents


def test_similarity_matrix_properties(model):
    sim = model.similarity
    assert sim.shape == (len(model.docs),) * 2
    assert abs(sim - sim.T).max() < 1e-9
    assert abs(sim.diagonal() - 1).max() < 1e-9


def test_classification_has_no_train_test_overlap(model):
    result = run_classification(model, cv_folds=0)
    assert not set(result["train_idx"]) & set(result["test_idx"])
    assert len(result["y_true"]) == len(result["y_pred"]) == len(result["test_idx"])
    assert 0 <= result["metrics"]["accuracy"] <= 1


def test_clustering_on_real_data(model):
    result = run_clustering(model, k=5, n_init=3)
    assert len(result["clusters"]) == model.eval_mask.sum()
    assert 0 < result["purity"] <= 1 and 0 < result["rand_index"] <= 1


def test_recommendation_excludes_self(model):
    recs, _ = recommend(0, model.similarity, model.docs, model.tfidf.vectors, top_k=5)
    assert len(recs) == 5 and model.docs["doc_id"].iloc[0] not in set(recs["doc_id"])
    assert recs["similarity"].is_monotonic_decreasing
