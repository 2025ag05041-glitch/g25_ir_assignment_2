"""Manual TF-IDF feature construction and cosine similarity.

No library TF-IDF is used. The weighting scheme is:

    tf weight   w_tf(t, d) = 1 + log10(tf(t, d))   if tf > 0   ("log" scheme)
                           = tf(t, d)                          ("raw" scheme)
    idf         idf(t)     = log10(N / df(t))
    tf-idf      w(t, d)    = w_tf(t, d) * idf(t)
    length norm w'(t, d)   = w(t, d) / sqrt(sum_t w(t, d)^2)    (L2 / cosine)

After L2 normalisation the cosine similarity of two documents is simply the
dot product of their vectors.
"""
import math
from collections import Counter

import numpy as np


def term_frequencies(docs_terms):
    """One Counter of raw term counts per document."""
    return [Counter(terms) for terms in docs_terms]


def document_frequencies(tf_list):
    """df(t) = number of documents containing term t."""
    df = Counter()
    for tf in tf_list:
        df.update(tf.keys())
    return df


def build_vocabulary(df, n_docs, min_df=1, max_df_ratio=1.0, max_features=None):
    """Keep terms with min_df <= df(t) <= max_df_ratio * N.

    Returns a dict term -> column index, sorted alphabetically so the
    column order is stable between runs.
    """
    max_df = max_df_ratio * n_docs
    terms = [t for t, d in df.items() if min_df <= d <= max_df]
    if max_features and len(terms) > max_features:
        terms = sorted(terms, key=lambda t: (-df[t], t))[:max_features]
    return {t: i for i, t in enumerate(sorted(terms))}


def compute_idf(df, n_docs, vocabulary):
    return {t: math.log10(n_docs / df[t]) for t in vocabulary}


def tf_weight(count, scheme="log"):
    if count <= 0:
        return 0.0
    return 1.0 + math.log10(count) if scheme == "log" else float(count)


def l2_normalize(vector):
    norm = math.sqrt(sum(w * w for w in vector.values()))
    if norm == 0:
        return dict(vector)
    return {t: w / norm for t, w in vector.items()}


def tfidf_vector(tf, idf, scheme="log", normalize=True):
    """Sparse TF-IDF vector {term: weight} for one document's term counts."""
    vec = {}
    for term, count in tf.items():
        if term in idf:
            weight = tf_weight(count, scheme) * idf[term]
            if weight != 0:
                vec[term] = weight
    return l2_normalize(vec) if normalize else vec


def cosine_similarity(vec_a, vec_b):
    """Cosine similarity of two sparse vectors (dicts)."""
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    dot = sum(w * vec_b.get(t, 0.0) for t, w in vec_a.items())
    norm_a = math.sqrt(sum(w * w for w in vec_a.values()))
    norm_b = math.sqrt(sum(w * w for w in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_similarity_matrix(matrix):
    """Pairwise cosine similarity for the rows of a dense matrix."""
    norms = np.sqrt((matrix * matrix).sum(axis=1))
    norms[norms == 0] = 1.0
    unit = matrix / norms[:, None]
    return unit @ unit.T


class TfidfModel:
    """Fits vocabulary, df and idf on a corpus and stores the document vectors."""

    def __init__(self, min_df=1, max_df_ratio=1.0, max_features=None,
                 tf_scheme="log", normalize=True):
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.max_features = max_features
        self.tf_scheme = tf_scheme
        self.normalize = normalize

    def fit(self, docs_terms):
        self.n_docs = len(docs_terms)
        self.tf_list = term_frequencies(docs_terms)
        self.df = document_frequencies(self.tf_list)
        self.vocabulary = build_vocabulary(self.df, self.n_docs, self.min_df,
                                           self.max_df_ratio, self.max_features)
        self.terms = sorted(self.vocabulary, key=self.vocabulary.get)
        self.idf = compute_idf(self.df, self.n_docs, self.vocabulary)
        self.vectors = [tfidf_vector(tf, self.idf, self.tf_scheme, self.normalize)
                        for tf in self.tf_list]
        self.matrix = self.to_matrix(self.vectors)
        return self

    def transform(self, terms):
        """TF-IDF vector for a new document/query using the fitted idf."""
        return tfidf_vector(Counter(terms), self.idf, self.tf_scheme, self.normalize)

    def to_matrix(self, vectors):
        matrix = np.zeros((len(vectors), len(self.vocabulary)))
        for row, vec in enumerate(vectors):
            for term, weight in vec.items():
                matrix[row, self.vocabulary[term]] = weight
        return matrix

    def top_terms(self, doc_index, n=10):
        vec = self.vectors[doc_index]
        return sorted(vec.items(), key=lambda kv: -kv[1])[:n]
