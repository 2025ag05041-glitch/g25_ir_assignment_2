"""Inverted index built from the manual TF-IDF vectors.

postings[term] = [(doc_index, raw_tf, tfidf_weight), ...] sorted by doc_index.
Query scoring is term-at-a-time: for every query term we walk its postings
list and add query_weight * doc_weight to the document's accumulator.
With L2-normalised vectors the accumulated score is the cosine similarity.
"""
from collections import defaultdict


class InvertedIndex:
    def __init__(self):
        self.postings = {}
        self.n_docs = 0

    def build(self, tf_list, vectors):
        postings = defaultdict(list)
        for doc_index, (tf, vec) in enumerate(zip(tf_list, vectors)):
            for term, weight in vec.items():
                postings[term].append((doc_index, tf[term], weight))
        self.postings = dict(postings)
        self.n_docs = len(vectors)
        return self

    def lookup(self, term):
        return self.postings.get(term, [])

    def document_frequency(self, term):
        return len(self.postings.get(term, []))

    def search(self, query_vector, top_k=10, exclude=()):
        scores = defaultdict(float)
        for term, q_weight in query_vector.items():
            for doc_index, _, d_weight in self.postings.get(term, []):
                scores[doc_index] += q_weight * d_weight
        ranked = sorted(((d, s) for d, s in scores.items() if d not in exclude),
                        key=lambda ds: (-ds[1], ds[0]))
        return ranked[:top_k]

    def stats(self):
        lengths = [len(p) for p in self.postings.values()]
        return {
            "terms": len(lengths),
            "postings": sum(lengths),
            "avg_postings_length": sum(lengths) / len(lengths) if lengths else 0.0,
            "max_postings_length": max(lengths) if lengths else 0,
        }
