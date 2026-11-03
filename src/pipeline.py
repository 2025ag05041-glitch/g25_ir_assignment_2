"""Glue code: runs preprocessing, TF-IDF, indexing, classification and
clustering for a given corpus and settings. Used by the Streamlit pages and
by the experiment script so both produce the same numbers."""
from dataclasses import dataclass

import numpy as np

from config import DEFAULT_MIN_DF, NB_ALPHA, RANDOM_SEED, TEST_SIZE
from src.classification import cross_validate, stratified_split, train_and_evaluate
from src.clustering import KMeans, centroid_top_terms, project_2d
from src.evaluation import contingency_table, pair_counts, purity, rand_index
from src.features import TfidfModel, cosine_similarity_matrix
from src.ingestion import is_evaluation_document
from src.inverted_index import InvertedIndex
from src.preprocessing import PreprocessConfig, preprocess


@dataclass(frozen=True)
class FeatureConfig:
    min_df: int = DEFAULT_MIN_DF
    max_df_ratio: float = 1.0
    tf_scheme: str = "log"      # "log" or "raw"
    normalize: bool = True      # L2 document-length normalisation


class CorpusModel:
    """Everything derived from one corpus + one preprocessing setting."""

    def __init__(self, docs, prep_config=PreprocessConfig(),
                 feature_config=FeatureConfig()):
        self.docs = docs.reset_index(drop=True)
        self.prep_config = prep_config
        self.feature_config = feature_config
        self.docs_terms = [preprocess(t, prep_config) for t in self.docs["text"]]
        self.tfidf = TfidfModel(min_df=feature_config.min_df,
                                max_df_ratio=feature_config.max_df_ratio,
                                tf_scheme=feature_config.tf_scheme,
                                normalize=feature_config.normalize).fit(self.docs_terms)
        self.matrix = self.tfidf.matrix
        self.similarity = cosine_similarity_matrix(self.matrix)
        self.index = InvertedIndex().build(self.tfidf.tf_list, self.tfidf.vectors)
        self.eval_mask = np.array([is_evaluation_document(c) for c in self.docs["category"]])

    @property
    def eval_indices(self):
        return np.where(self.eval_mask)[0].tolist()


def run_classification(model, test_size=TEST_SIZE, seed=RANDOM_SEED, alpha=NB_ALPHA,
                       groups=None, cv_folds=5):
    idx = model.eval_indices
    terms = [model.docs_terms[i] for i in idx]
    labels = model.docs["category"].iloc[idx].tolist()
    sub_groups = None if groups is None else [groups[i] for i in idx]
    train, test = stratified_split(labels, test_size, seed, sub_groups)
    nb, y_true, y_pred, metrics = train_and_evaluate(terms, labels, train, test, alpha)
    result = {
        "model": nb,
        "train_idx": [idx[i] for i in train],
        "test_idx": [idx[i] for i in test],
        "y_true": y_true,
        "y_pred": y_pred,
        "metrics": metrics,
    }
    if cv_folds:
        result["cv_metrics"] = cross_validate(terms, labels, cv_folds, alpha, seed, sub_groups)
    return result


def run_clustering(model, k=5, init="k-means++", n_init=10, max_iter=100,
                   seed=RANDOM_SEED):
    idx = model.eval_indices
    X = model.matrix[idx]
    km = KMeans(k=k, init=init, n_init=n_init, max_iter=max_iter, seed=seed).fit(X)
    clusters = km.labels_.tolist()
    labels = model.docs["category"].iloc[idx].tolist()
    coords, variance = project_2d(X)
    return {
        "kmeans": km,
        "indices": idx,
        "clusters": clusters,
        "labels": labels,
        "purity": purity(clusters, labels),
        "rand_index": rand_index(clusters, labels),
        "pair_counts": pair_counts(clusters, labels),
        "contingency": contingency_table(clusters, labels),
        "top_terms": centroid_top_terms(km.centroids_, model.tfidf.terms),
        "coords": coords,
        "explained_variance": variance,
    }
