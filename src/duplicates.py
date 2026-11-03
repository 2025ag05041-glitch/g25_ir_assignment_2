"""Exact and near-duplicate detection.

Exact duplicates: identical MD5 hash after normalising case, punctuation and
whitespace.
Near duplicates: Jaccard similarity of word 5-gram shingle sets
(J = |A n B| / |A u B|) and the TF-IDF cosine similarity. A pair is flagged
when either score passes its threshold.
"""
import hashlib
import re

import numpy as np
import pandas as pd

WORD_RE = re.compile(r"[^\W_]+")


def normalized_hash(text):
    words = WORD_RE.findall(text.lower())
    return hashlib.md5(" ".join(words).encode("utf-8")).hexdigest()


def shingles(text, k=5):
    words = WORD_RE.findall(text.lower())
    if len(words) < k:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def pairwise_similarity(df, sim_matrix=None, k=5):
    """Jaccard (and cosine, if given) for every document pair."""
    texts = df["text"].tolist()
    ids = df["doc_id"].tolist()
    shingle_sets = [shingles(t, k) for t in texts]
    hashes = [normalized_hash(t) for t in texts]
    rows = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            rows.append({
                "doc_a": ids[i], "doc_b": ids[j],
                "title_a": df["title"].iloc[i], "title_b": df["title"].iloc[j],
                "exact_match": hashes[i] == hashes[j],
                "jaccard": jaccard(shingle_sets[i], shingle_sets[j]),
                "cosine": float(sim_matrix[i, j]) if sim_matrix is not None else np.nan,
                "i": i, "j": j,
            })
    return pd.DataFrame(rows)


def classify_pairs(pairs, jaccard_threshold=0.5, cosine_threshold=0.9):
    def kind(row):
        if row["exact_match"]:
            return "Exact duplicate"
        if row["jaccard"] >= jaccard_threshold or row["cosine"] >= cosine_threshold:
            return "Near duplicate"
        return "Distinct"
    pairs = pairs.copy()
    pairs["type"] = pairs.apply(kind, axis=1) if len(pairs) else []
    return pairs


def duplicate_groups(n_docs, flagged_pairs):
    """Union-find over flagged pairs; returns a group id per document."""
    parent = list(range(n_docs))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in flagged_pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return [find(i) for i in range(n_docs)]


def make_near_duplicate(text, change_rate=0.1, seed=0):
    """Create an edited copy of a text by deleting a fraction of its words.
    Used only for the clearly labelled duplicate-injection experiment."""
    rng = np.random.RandomState(seed)
    words = text.split()
    keep = rng.rand(len(words)) >= change_rate
    return " ".join(w for w, k in zip(words, keep) if k)
