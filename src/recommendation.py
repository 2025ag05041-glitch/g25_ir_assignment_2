"""Content-based Top-K document recommendation.

A document is represented by its L2-normalised TF-IDF vector. The related
documents are the K documents with the highest cosine similarity to it,
excluding the document itself and (optionally) its near-duplicates.
"""
import numpy as np
import pandas as pd


def shared_terms(vec_a, vec_b, n=6):
    """Terms that contribute most to the cosine score (w_a * w_b)."""
    common = [(t, vec_a[t] * vec_b[t]) for t in vec_a.keys() & vec_b.keys()]
    common.sort(key=lambda x: -x[1])
    return [t for t, _ in common[:n]]


def recommend(doc_index, sim_matrix, docs, vectors, top_k=5,
              exclude_duplicates=True, duplicate_threshold=0.9):
    scores = sim_matrix[doc_index]
    order = np.argsort(-scores, kind="stable")
    source_category = docs["category"].iloc[doc_index]
    rows, skipped = [], []
    for j in order:
        if j == doc_index:
            continue
        if exclude_duplicates and scores[j] >= duplicate_threshold:
            skipped.append(docs["doc_id"].iloc[j])
            continue
        rows.append({
            "rank": len(rows) + 1,
            "doc_id": docs["doc_id"].iloc[j],
            "title": docs["title"].iloc[j],
            "category": docs["category"].iloc[j],
            "similarity": float(scores[j]),
            "same_category": docs["category"].iloc[j] == source_category,
            "shared_terms": ", ".join(shared_terms(vectors[doc_index], vectors[j])),
            "index": int(j),
        })
        if len(rows) == top_k:
            break
    return pd.DataFrame(rows), skipped


def recommend_for_query(query_vector, index, docs, vectors, top_k=5):
    """Recommend documents for free text using the inverted index."""
    results = index.search(query_vector, top_k=top_k)
    rows = []
    for rank, (j, score) in enumerate(results, start=1):
        rows.append({
            "rank": rank,
            "doc_id": docs["doc_id"].iloc[j],
            "title": docs["title"].iloc[j],
            "category": docs["category"].iloc[j],
            "similarity": score,
            "shared_terms": ", ".join(shared_terms(query_vector, vectors[j])),
            "index": int(j),
        })
    return pd.DataFrame(rows)


def category_precision_at_k(sim_matrix, categories, k=5, mask=None):
    """Average share of the top-k recommendations that have the same category
    as the source document. Uses only documents where mask is True (labelled
    documents) as sources and candidates."""
    categories = list(categories)
    n = len(categories)
    allowed = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    candidates = np.where(allowed)[0]
    precisions = []
    for i in candidates:
        others = [j for j in candidates if j != i]
        if not others:
            continue
        ranked = sorted(others, key=lambda j: (-sim_matrix[i, j], j))[:k]
        precisions.append(sum(categories[j] == categories[i] for j in ranked) / len(ranked))
    return float(np.mean(precisions)) if precisions else 0.0
