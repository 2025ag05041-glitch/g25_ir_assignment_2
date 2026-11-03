"""Unit tests for the manually implemented algorithms.

Expected values are calculated by hand or taken from the worked examples in
Manning, Raghavan & Schuetze, "Introduction to Information Retrieval" (IIR).
Run with:  python -m pytest -q
"""
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification import (MultinomialNaiveBayes, cross_validate,  # noqa: E402
                                stratified_split)
from src.clustering import KMeans, project_2d  # noqa: E402
from src.evaluation import (classification_metrics, confusion_matrix, pair_counts,  # noqa: E402
                            purity, rand_index, rand_index_bruteforce)
from src.features import (TfidfModel, build_vocabulary, cosine_similarity,  # noqa: E402
                          document_frequencies, term_frequencies)
from src.inverted_index import InvertedIndex  # noqa: E402


# ------------------------------------------------------------------ TF-IDF
DOCS = [["car", "insurance", "car"], ["car", "auto"], ["insurance", "claim", "claim", "claim"]]


def test_term_and_document_frequencies():
    tf = term_frequencies(DOCS)
    assert tf[0]["car"] == 2 and tf[2]["claim"] == 3
    df = document_frequencies(tf)
    assert df == {"car": 2, "insurance": 2, "auto": 1, "claim": 1}


def test_tfidf_values_match_hand_calculation():
    model = TfidfModel(min_df=1, tf_scheme="log", normalize=False).fit(DOCS)
    idf_car = math.log10(3 / 2)
    idf_auto = math.log10(3 / 1)
    assert model.idf["car"] == pytest.approx(idf_car)
    # doc 0: tf(car)=2 -> (1 + log10 2) * idf
    assert model.vectors[0]["car"] == pytest.approx((1 + math.log10(2)) * idf_car)
    assert model.vectors[1]["auto"] == pytest.approx(idf_auto)


def test_l2_normalisation_gives_unit_vectors():
    model = TfidfModel(min_df=1).fit(DOCS)
    for vec in model.vectors:
        assert math.sqrt(sum(w * w for w in vec.values())) == pytest.approx(1.0)
    assert np.allclose(np.linalg.norm(model.matrix, axis=1), 1.0)


def test_term_in_every_document_gets_zero_idf():
    model = TfidfModel(min_df=1).fit([["a", "b"], ["a", "c"]])
    assert model.idf["a"] == 0
    assert "a" not in model.vectors[0]


def test_min_df_filters_vocabulary():
    df = document_frequencies(term_frequencies(DOCS))
    vocab = build_vocabulary(df, 3, min_df=2)
    assert set(vocab) == {"car", "insurance"}


def test_raw_tf_scheme():
    model = TfidfModel(min_df=1, tf_scheme="raw", normalize=False).fit(DOCS)
    assert model.vectors[2]["claim"] == pytest.approx(3 * math.log10(3))


def test_cosine_similarity():
    assert cosine_similarity({"a": 1, "b": 0}, {"a": 2}) == pytest.approx(1.0)
    assert cosine_similarity({"a": 1}, {"b": 1}) == 0.0
    assert cosine_similarity({"a": 1, "b": 1}, {"a": 1}) == pytest.approx(1 / math.sqrt(2))
    assert cosine_similarity({}, {"a": 1}) == 0.0


def test_inverted_index_postings_and_search_equal_cosine():
    model = TfidfModel(min_df=1).fit(DOCS)
    index = InvertedIndex().build(model.tf_list, model.vectors)
    assert [p[0] for p in index.lookup("car")] == [0, 1]
    assert index.lookup("car")[0][1] == 2          # raw tf stored in postings
    assert index.lookup("missing") == []
    query = model.transform(["car", "insurance"])
    results = dict(index.search(query, top_k=3))
    for doc, score in results.items():
        assert score == pytest.approx(cosine_similarity(query, model.vectors[doc]))


# ------------------------------------------------------------- Naive Bayes
def test_naive_bayes_iir_example_13_1():
    """IIR Example 13.1: the test document is classified as 'china'."""
    train = [["chinese", "beijing", "chinese"], ["chinese", "chinese", "shanghai"],
             ["chinese", "macao"], ["tokyo", "japan", "chinese"]]
    labels = ["china", "china", "china", "japan"]
    nb = MultinomialNaiveBayes(alpha=1.0).fit(train, labels)
    assert math.exp(nb.log_prior["china"]) == pytest.approx(3 / 4)
    assert math.exp(nb.log_likelihood["china"]["chinese"]) == pytest.approx(6 / 14)
    assert math.exp(nb.log_unseen["china"]) == pytest.approx(1 / 14)        # P(tokyo|china)
    assert math.exp(nb.log_likelihood["japan"]["chinese"]) == pytest.approx(2 / 9)
    test = ["chinese", "chinese", "chinese", "tokyo", "japan"]
    scores = nb.class_scores(test)
    assert math.exp(scores["china"]) == pytest.approx(3 / 4 * (3 / 7) ** 3 * (1 / 14) ** 2)
    assert math.exp(scores["japan"]) == pytest.approx(1 / 4 * (2 / 9) ** 3 * (2 / 9) ** 2)
    assert nb.predict_one(test) == "china"
    assert sum(nb.posterior(test).values()) == pytest.approx(1.0)


def test_naive_bayes_ignores_unknown_terms():
    nb = MultinomialNaiveBayes().fit([["a"], ["b"]], ["x", "y"])
    assert nb.class_scores(["zzz"]) == nb.class_scores([])


def test_naive_bayes_rejects_bad_alpha():
    with pytest.raises(ValueError):
        MultinomialNaiveBayes(alpha=0)


# --------------------------------------------------------------- metrics
def test_classification_metrics_hand_example():
    y_true = ["a", "a", "a", "b", "b", "c"]
    y_pred = ["a", "a", "b", "b", "c", "c"]
    m = classification_metrics(y_true, y_pred, ["a", "b", "c"])
    assert m["confusion_matrix"] == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]
    assert m["accuracy"] == pytest.approx(4 / 6)
    a, b, c = m["per_class"]
    assert (a["precision"], a["recall"]) == (1.0, pytest.approx(2 / 3))
    assert (b["precision"], b["recall"]) == (0.5, 0.5)
    assert (c["precision"], c["recall"]) == (0.5, 1.0)
    assert a["f1"] == pytest.approx(0.8)
    assert m["macro_precision"] == pytest.approx((1 + 0.5 + 0.5) / 3)
    assert m["weighted_recall"] == pytest.approx((3 * 2 / 3 + 2 * 0.5 + 1 * 1) / 6)


def test_metrics_zero_division_is_zero():
    m = classification_metrics(["a", "a"], ["a", "a"], ["a", "b"])
    assert m["per_class"][1]["precision"] == 0.0 and m["per_class"][1]["f1"] == 0.0


def test_confusion_matrix_binary():
    assert confusion_matrix([1, 1, 0, 0], [1, 0, 0, 1], [0, 1]) == [[1, 1], [1, 1]]


# IIR Figure 16.4: three clusters of x / o / d documents
IIR_CLUSTERS = [1] * 6 + [2] * 6 + [3] * 5
IIR_LABELS = (["x"] * 5 + ["o"]) + (["x"] + ["o"] * 4 + ["d"]) + (["x"] * 2 + ["d"] * 3)


def test_purity_iir_example():
    assert purity(IIR_CLUSTERS, IIR_LABELS) == pytest.approx((5 + 4 + 3) / 17)


def test_rand_index_iir_example():
    assert pair_counts(IIR_CLUSTERS, IIR_LABELS) == {"TP": 20, "FP": 20, "FN": 24, "TN": 72}
    assert rand_index(IIR_CLUSTERS, IIR_LABELS) == pytest.approx(92 / 136)


def test_rand_index_matches_bruteforce_on_random_data():
    rng = random.Random(0)
    for _ in range(20):
        n = rng.randint(2, 40)
        clusters = [rng.randint(0, 4) for _ in range(n)]
        labels = [rng.choice("abc") for _ in range(n)]
        assert rand_index(clusters, labels) == pytest.approx(rand_index_bruteforce(clusters, labels))


def test_purity_and_rand_index_edge_cases():
    assert purity([0, 0, 1, 1], ["a", "a", "b", "b"]) == 1.0
    assert rand_index([0, 0, 1, 1], ["a", "a", "b", "b"]) == 1.0
    assert purity([0, 0, 0, 0], ["a", "a", "b", "b"]) == 0.5
    assert rand_index([0], ["a"]) == 1.0              # no pairs
    assert purity([], []) == 0.0
    with pytest.raises(ValueError):
        purity([0, 1], ["a"])
    # label names do not matter, only the grouping
    assert rand_index([5, 5, 9], ["x", "x", "y"]) == rand_index([0, 0, 1], ["x", "x", "y"])


# ---------------------------------------------------------------- K-Means
def test_kmeans_separates_two_obvious_groups():
    rng = np.random.RandomState(1)
    X = np.vstack([rng.normal(0, 0.1, (10, 3)), rng.normal(5, 0.1, (10, 3))])
    km = KMeans(k=2, n_init=3, seed=0).fit(X)
    assert len(set(km.labels_[:10])) == 1 and len(set(km.labels_[10:])) == 1
    assert km.labels_[0] != km.labels_[10]
    assert km.converged_
    assert np.allclose(sorted(km.centroids_[:, 0]), [X[:10, 0].mean(), X[10:, 0].mean()])


def test_kmeans_sse_never_increases():
    X = np.random.RandomState(3).rand(40, 5)
    km = KMeans(k=4, init="random", n_init=1, seed=3).fit(X)
    sse = [h["sse"] for h in km.history_]
    assert all(b <= a + 1e-9 for a, b in zip(sse, sse[1:]))


def test_kmeans_handles_identical_points_and_empty_clusters():
    X = np.ones((6, 3))
    km = KMeans(k=3, n_init=2, seed=0).fit(X)
    assert len(km.labels_) == 6 and km.sse_ == pytest.approx(0.0)


def test_kmeans_too_few_documents():
    with pytest.raises(ValueError):
        KMeans(k=5).fit(np.ones((3, 2)))


def test_kmeans_is_reproducible():
    X = np.random.RandomState(5).rand(30, 4)
    a = KMeans(k=3, seed=7).fit(X).labels_
    b = KMeans(k=3, seed=7).fit(X).labels_
    assert (a == b).all()


def test_project_2d_shape():
    coords, var = project_2d(np.random.RandomState(0).rand(10, 6))
    assert coords.shape == (10, 2) and len(var) == 2


# ------------------------------------------------------------- splitting
def test_stratified_split_keeps_groups_together():
    labels = ["a"] * 10 + ["b"] * 10
    groups = [i // 2 for i in range(20)]          # pairs of duplicates
    train, test = stratified_split(labels, 0.3, 1, groups)
    assert not set(train) & set(test)
    assert sorted(train + test) == list(range(20))
    for g in set(groups):
        members = [i for i in range(20) if groups[i] == g]
        assert all(m in train for m in members) or all(m in test for m in members)
    assert {labels[i] for i in test} == {"a", "b"}


def test_cross_validation_runs():
    docs = [["x", "y"]] * 5 + [["p", "q"]] * 5
    labels = ["A"] * 5 + ["B"] * 5
    m = cross_validate(docs, labels, k=5)
    assert m["accuracy"] == 1.0 and len(m["fold_accuracy"]) == 5


def test_no_prohibited_library_implementations():
    """TF-IDF, Naive Bayes, K-Means, Purity and Rand Index must be our own code."""
    src = Path(__file__).resolve().parents[1] / "src"
    for path in src.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "sklearn" not in text and "scikit" not in text, path.name


def test_split_does_not_depend_on_group_numbering():
    labels = ["a"] * 12 + ["b"] * 9
    plain = stratified_split(labels, 0.3, 42)
    singleton_groups = [i + 50 for i in range(len(labels))]   # e.g. global document indices
    assert stratified_split(labels, 0.3, 42, singleton_groups) == plain
