"""Experiments used for the inference questions in the report.

1. compare_preprocessing  - effect of preprocessing choices on vocabulary,
                            Naive Bayes, K-Means and recommendations.
2. duplicate_experiment   - injects clearly labelled copies of some documents
                            (in memory only) to measure how duplicates affect
                            the results, with and without our handling.
3. k_sweep                - Purity / Rand Index / SSE for different K.
"""
from collections import Counter

import numpy as np
import pandas as pd

from config import NEAR_DUPLICATE_COSINE, NEAR_DUPLICATE_JACCARD, RANDOM_SEED
from src.classification import stratified_split, train_and_evaluate
from src.clustering import KMeans
from src.duplicates import (classify_pairs, duplicate_groups, make_near_duplicate,
                            pairwise_similarity)
from src.evaluation import purity, rand_index
from src.ingestion import add_length_columns
from src.pipeline import CorpusModel, FeatureConfig, run_classification, run_clustering
from src.preprocessing import PreprocessConfig
from src.recommendation import category_precision_at_k

CLUSTER_SEEDS = [42, 1, 2, 3, 4]

PREPROCESSING_VARIANTS = {
    "A. Minimal (lowercase + tokens only)": (
        PreprocessConfig(remove_noise=False, strip_references=False,
                         remove_stopwords=False, remove_numbers=False,
                         normalizer="none"), FeatureConfig()),
    "B. A + noise/reference/number cleaning": (
        PreprocessConfig(remove_stopwords=False, normalizer="none"), FeatureConfig()),
    "C. B + stop-word removal": (
        PreprocessConfig(normalizer="none"), FeatureConfig()),
    "D. C + Porter stemming (default)": (
        PreprocessConfig(normalizer="stem"), FeatureConfig()),
    "E. C + lemmatization": (
        PreprocessConfig(normalizer="lemma"), FeatureConfig()),
    "F. D + bigrams": (
        PreprocessConfig(normalizer="stem", ngram_max=2), FeatureConfig()),
    "G. D without length normalization": (
        PreprocessConfig(normalizer="stem"), FeatureConfig(normalize=False)),
}


def evaluate_setting(docs, prep, feat, k=5, seed=RANDOM_SEED):
    model = CorpusModel(docs, prep, feat)
    vocab = Counter()
    for terms in model.docs_terms:
        vocab.update(terms)
    clf = run_classification(model, seed=seed)
    # K-Means depends on the initial seeds, so average over five seeds
    clusterings = [run_clustering(model, k=k, seed=s) for s in CLUSTER_SEEDS]
    return {
        "vocabulary_size": len(vocab),
        "tfidf_features": len(model.tfidf.vocabulary),
        "avg_terms_per_doc": float(np.mean([len(t) for t in model.docs_terms])),
        "nb_accuracy": clf["metrics"]["accuracy"],
        "nb_macro_f1": clf["metrics"]["macro_f1"],
        "nb_cv_accuracy": clf["cv_metrics"]["accuracy"],
        "nb_cv_macro_f1": clf["cv_metrics"]["macro_f1"],
        "kmeans_purity": float(np.mean([c["purity"] for c in clusterings])),
        "kmeans_rand_index": float(np.mean([c["rand_index"] for c in clusterings])),
        "recommendation_p_at_5": category_precision_at_k(
            model.similarity, model.docs["category"], 5, model.eval_mask),
    }


def compare_preprocessing(docs, variants=None, progress=None):
    variants = variants or PREPROCESSING_VARIANTS
    rows = []
    for n, (name, (prep, feat)) in enumerate(variants.items(), start=1):
        row = {"setting": name}
        row.update(evaluate_setting(docs, prep, feat))
        rows.append(row)
        if progress:
            progress(n / len(variants), name)
    return pd.DataFrame(rows)


def k_sweep(model, k_values=range(2, 11), seed=RANDOM_SEED):
    rows = []
    for k in k_values:
        result = run_clustering(model, k=k, seed=seed)
        rows.append({"k": k, "purity": result["purity"],
                     "rand_index": result["rand_index"],
                     "sse": result["kmeans"].sse_})
    return pd.DataFrame(rows)


def inject_duplicates(docs, n_sources=10, change_rate=0.1, seed=RANDOM_SEED):
    """Return a copy of the corpus with one exact copy and one edited copy
    (change_rate of the words deleted) of n_sources labelled documents.
    The copies are marked [SIMULATED] and exist only in memory."""
    rng = np.random.RandomState(seed)
    labelled = docs[docs["category"].isin(
        [c for c in docs["category"].unique() if c != "General (Off-domain)"])]
    sources = labelled.sample(n=min(n_sources, len(labelled)), random_state=rng)
    extra = []
    for n, (_, src) in enumerate(sources.iterrows(), start=1):
        for kind, text in (("exact", src["text"]),
                           ("near", make_near_duplicate(src["text"], change_rate, seed + n))):
            row = src.to_dict()
            row.update({"doc_id": f"SIM{n:02d}{kind[0].upper()}",
                        "title": f"[SIMULATED {kind} copy] {src['title']}",
                        "source": f"Simulated {kind} copy of {src['doc_id']}",
                        "text": text, "copy_of": src["doc_id"]})
            extra.append(row)
    augmented = pd.concat([docs.assign(copy_of=""), pd.DataFrame(extra)],
                          ignore_index=True)
    return add_length_columns(augmented), sources["doc_id"].tolist()


def duplicate_experiment(docs, n_sources=10, change_rate=0.1, n_splits=20,
                         seed=RANDOM_SEED, k=5):
    prep, feat = PreprocessConfig(), FeatureConfig()
    augmented, source_ids = inject_duplicates(docs, n_sources, change_rate, seed)
    original_model = CorpusModel(docs, prep, feat)
    model = CorpusModel(augmented, prep, feat)
    ids = model.docs["doc_id"].tolist()

    # 1. detection
    pairs = classify_pairs(pairwise_similarity(model.docs, model.similarity),
                           NEAR_DUPLICATE_JACCARD, NEAR_DUPLICATE_COSINE)
    flagged = pairs[pairs["type"] != "Distinct"]
    copy_of = dict(zip(ids, model.docs["copy_of"]))

    def root(doc_id):
        return copy_of.get(doc_id) or doc_id

    true_pairs = {tuple(sorted((a, b))) for a in ids for b in ids
                  if a < b and root(a) == root(b)}
    found = {tuple(sorted((a, b))) for a, b in zip(flagged["doc_a"], flagged["doc_b"])}
    detection = {
        "injected_pairs": len(true_pairs),
        "detected_injected_pairs": len(true_pairs & found),
        "false_positive_pairs": len(found - true_pairs),
        "exact_pairs_flagged": int((flagged["type"] == "Exact duplicate").sum()),
        "near_pairs_flagged": int((flagged["type"] == "Near duplicate").sum()),
        "min_jaccard_near_copy": float(flagged.loc[flagged["type"] == "Near duplicate",
                                                   "jaccard"].min()) if len(flagged) else 0.0,
    }
    groups = duplicate_groups(len(ids), [(int(i), int(j)) for i, j in
                                         zip(flagged["i"], flagged["j"])])

    # 2. classification: leaky random split vs group-aware split
    idx = model.eval_indices
    terms = [model.docs_terms[i] for i in idx]
    labels = model.docs["category"].iloc[idx].tolist()
    sub_groups = [groups[i] for i in idx]
    o_idx = original_model.eval_indices
    o_terms = [original_model.docs_terms[i] for i in o_idx]
    o_labels = original_model.docs["category"].iloc[o_idx].tolist()
    acc = {"original_corpus": [], "with_duplicates_naive_split": [],
           "with_duplicates_group_split": []}
    leaked = []
    for s in range(n_splits):
        tr, te = stratified_split(o_labels, 0.3, s)
        acc["original_corpus"].append(train_and_evaluate(o_terms, o_labels, tr, te)[3]["accuracy"])
        tr, te = stratified_split(labels, 0.3, s)
        acc["with_duplicates_naive_split"].append(
            train_and_evaluate(terms, labels, tr, te)[3]["accuracy"])
        train_groups = {sub_groups[i] for i in tr}
        leaked.append(sum(1 for i in te if sub_groups[i] in train_groups
                          and len([g for g in sub_groups if g == sub_groups[i]]) > 1))
        tr, te = stratified_split(labels, 0.3, s, sub_groups)
        acc["with_duplicates_group_split"].append(
            train_and_evaluate(terms, labels, tr, te)[3]["accuracy"])
    classification = {name: float(np.mean(v)) for name, v in acc.items()}
    classification["avg_leaked_test_docs_naive_split"] = float(np.mean(leaked))
    classification["splits"] = n_splits

    # 3. recommendation: how many top-5 slots are taken by copies
    sim = model.similarity
    source_rows = [ids.index(d) for d in source_ids]
    group_of = {i: groups[i] for i in range(len(ids))}

    def wasted_slots(filter_duplicates):
        wasted = 0
        for i in source_rows:
            order = [j for j in np.argsort(-sim[i]) if j != i]
            if filter_duplicates:
                order = [j for j in order if sim[i, j] < NEAR_DUPLICATE_COSINE
                         and group_of[j] != group_of[i]]
            wasted += sum(1 for j in order[:5] if group_of[j] == group_of[i])
        return wasted / (5 * len(source_rows))

    recommendation = {
        "share_of_top5_slots_taken_by_copies_unfiltered": wasted_slots(False),
        "share_of_top5_slots_taken_by_copies_filtered": wasted_slots(True),
        "p_at_5_original": category_precision_at_k(original_model.similarity,
                                                   original_model.docs["category"], 5,
                                                   original_model.eval_mask),
        "p_at_5_with_duplicates": category_precision_at_k(sim, model.docs["category"], 5,
                                                          model.eval_mask),
    }

    # 4. clustering
    orig = run_clustering(original_model, k=k, seed=seed)
    dup = run_clustering(model, k=k, seed=seed)
    cl = dict(zip([ids[i] for i in dup["indices"]], dup["clusters"]))
    together = sum(1 for d in ids if copy_of[d] and cl.get(d) == cl.get(copy_of[d]))
    # dedup: drop the copies again (keep one document per group) and re-cluster
    keep = sorted({min(i for i in idx if groups[i] == g) for g in {groups[i] for i in idx}})
    km = KMeans(k=k, seed=seed).fit(model.matrix[keep])
    keep_labels = model.docs["category"].iloc[keep].tolist()
    clustering = {
        "purity_original": orig["purity"], "rand_index_original": orig["rand_index"],
        "purity_with_duplicates": dup["purity"], "rand_index_with_duplicates": dup["rand_index"],
        "purity_after_dedup": purity(km.labels_.tolist(), keep_labels),
        "rand_index_after_dedup": rand_index(km.labels_.tolist(), keep_labels),
        "copies_in_same_cluster_as_source": together,
        "copies_total": int(sum(1 for d in ids if copy_of[d])),
    }
    return {"source_ids": source_ids, "detection": detection,
            "classification": classification, "recommendation": recommendation,
            "clustering": clustering, "n_documents_augmented": len(ids)}
