"""Run all experiments with the default settings and save results + figures.

Usage:  python scripts/run_experiments.py
Output: reports/results/*.csv|json, reports/figures/*.png,
        data/processed/corpus_metadata.csv
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import CATEGORY_COLORS, FIGURES_DIR, RESULTS_DIR, SERIES_COLORS  # noqa: E402
from src.duplicates import classify_pairs, pairwise_similarity  # noqa: E402
from src.experiments import compare_preprocessing, duplicate_experiment, k_sweep  # noqa: E402
from src.ingestion import corpus_statistics, load_base_corpus, save_metadata  # noqa: E402
from src.pipeline import CorpusModel, run_classification, run_clustering  # noqa: E402
from src.preprocessing import raw_tokens, vocabulary_stats  # noqa: E402
from src.recommendation import category_precision_at_k, recommend, recommend_for_query  # noqa: E402

SHORT = {"Machine Learning": "ML", "Language & Search Technology": "Lang&Search",
         "AI Systems & Cognition": "AI Systems", "Computing & Infrastructure": "Computing",
         "Finance & Economics": "Finance", "General (Off-domain)": "Off-domain"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "axes.axisbelow": True,
                     "figure.dpi": 150})


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)


def heatmap(ax, z, xlabels, ylabels):
    z = np.array(z)
    ax.imshow(z, cmap="Blues")
    ax.grid(False)
    for (r, c), v in np.ndenumerate(z):
        ax.text(c, r, str(v), ha="center", va="center",
                color="white" if v > z.max() * 0.6 else "#0b0b0b")
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=30, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_base_corpus()
    save_metadata(docs)
    stats = corpus_statistics(docs)
    print("Corpus:", stats)

    model = CorpusModel(docs)
    raw = vocabulary_stats([raw_tokens(t) for t in docs["text"]])
    processed = vocabulary_stats(model.docs_terms)
    clf = run_classification(model)
    clu = run_clustering(model)
    sweep = k_sweep(model)
    pairs = classify_pairs(pairwise_similarity(docs, model.similarity))
    m = clf["metrics"]
    labels = m["labels"]

    # ---------------- recommendation examples
    examples = {}
    for title in ["Long short-term memory", "Stock market", "Agent (economics)",
                  "Information retrieval", "Cloud computing"]:
        i = int(docs.index[docs["title"] == title][0])
        recs, _ = recommend(i, model.similarity, docs, model.tfidf.vectors, 5)
        examples[title] = {"doc_id": docs["doc_id"].iloc[i], "category": docs["category"].iloc[i],
                           "recommendations": recs.drop(columns=["index"]).to_dict("records")}
    query = "How do recurrent neural networks handle long sequences in speech recognition?"
    from src.preprocessing import preprocess
    q_res = recommend_for_query(model.tfidf.transform(preprocess(query, model.prep_config)),
                                model.index, docs, model.tfidf.vectors, 5)

    km = clu["kmeans"]
    idx = clu["indices"]
    default_run = {
        "corpus": stats,
        "category_counts": docs["category"].value_counts().to_dict(),
        "vocabulary": {"raw_vocabulary": raw["vocabulary_size"], "raw_tokens": raw["total_tokens"],
                       "raw_avg_doc_length": raw["avg_doc_length"],
                       "processed_vocabulary": processed["vocabulary_size"],
                       "processed_tokens": processed["total_tokens"],
                       "processed_avg_doc_length": processed["avg_doc_length"],
                       "tfidf_features": len(model.tfidf.vocabulary),
                       "index": model.index.stats(),
                       "top_raw_terms": raw["counts"].most_common(15),
                       "top_processed_terms": processed["counts"].most_common(15)},
        "classification": {
            "train_docs": len(clf["train_idx"]), "test_docs": len(clf["test_idx"]),
            "metrics": {k: v for k, v in m.items()},
            "cv_metrics": clf["cv_metrics"],
            "predictions": [{"doc_id": docs["doc_id"].iloc[i], "title": docs["title"].iloc[i],
                             "actual": a, "predicted": p}
                            for i, a, p in zip(clf["test_idx"], clf["y_true"], clf["y_pred"])],
            "top_terms": {c: [t for t, _ in v] for c, v in clf["model"].top_terms(8).items()},
        },
        "clustering": {
            "k": km.k, "purity": clu["purity"], "rand_index": clu["rand_index"],
            "pair_counts": clu["pair_counts"], "n_iter": km.n_iter_, "converged": km.converged_,
            "sse": km.sse_, "history": km.history_, "best_run": km.best_run,
            "initial_seeds": [docs["title"].iloc[idx[s]] for s in km.initial_seeds_],
            "sizes": np.bincount(clu["clusters"], minlength=km.k).tolist(),
            "top_terms": clu["top_terms"],
            "contingency": {str(c): dict(v) for c, v in clu["contingency"].items()},
            "membership": [{"doc_id": docs["doc_id"].iloc[i], "title": docs["title"].iloc[i],
                            "label": l, "cluster": c}
                           for i, l, c in zip(idx, clu["labels"], clu["clusters"])],
            "explained_variance_2d": clu["explained_variance"],
            "runs": [{"sse": r["sse"], "n_iter": r["n_iter"]} for r in km.runs],
        },
        "recommendation": {"p_at_5": category_precision_at_k(model.similarity, docs["category"], 5,
                                                             model.eval_mask),
                           "examples": examples, "query": query,
                           "query_results": q_res.drop(columns=["index"]).to_dict("records")},
        "duplicates": {"pairs_compared": len(pairs),
                       "exact": int((pairs["type"] == "Exact duplicate").sum()),
                       "near": int((pairs["type"] == "Near duplicate").sum()),
                       "top_pairs": pairs.sort_values("jaccard", ascending=False).head(8)
                       [["doc_a", "title_a", "doc_b", "title_b", "jaccard", "cosine"]].to_dict("records"),
                       "max_cosine_pair": pairs.sort_values("cosine", ascending=False).head(5)
                       [["title_a", "title_b", "jaccard", "cosine"]].to_dict("records")},
    }
    (RESULTS_DIR / "default_run.json").write_text(json.dumps(default_run, indent=1, default=str))
    sweep.to_csv(RESULTS_DIR / "k_sweep.csv", index=False)

    print("Running preprocessing comparison...")
    prep_df = compare_preprocessing(docs)
    prep_df.to_csv(RESULTS_DIR / "preprocessing_comparison.csv", index=False)
    print(prep_df.round(3).to_string())

    print("Running duplicate experiment...")
    dup = duplicate_experiment(docs)
    (RESULTS_DIR / "duplicate_experiment.json").write_text(json.dumps(dup, indent=1, default=str))
    print(json.dumps(dup, indent=1, default=str))

    make_figures(docs, raw, processed, clf, clu, sweep, prep_df, examples)
    print("Done. Results in", RESULTS_DIR, "and figures in", FIGURES_DIR)


def make_figures(docs, raw, processed, clf, clu, sweep, prep_df, examples):
    m = clf["metrics"]
    labels = m["labels"]
    short = [SHORT[l] for l in labels]

    counts = docs["category"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.barh([SHORT[c] for c in counts.index], counts.values,
            color=[CATEGORY_COLORS[c] for c in counts.index])
    for y, v in enumerate(counts.values):
        ax.text(v + 0.3, y, str(v), va="center")
    ax.set_xlabel("Documents")
    ax.set_title("Documents per category")
    save(fig, "category_distribution.png")

    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.hist(docs["word_count"], bins=25, color=SERIES_COLORS[0], edgecolor="white")
    ax.set_xlabel("Words per document")
    ax.set_ylabel("Documents")
    ax.set_title("Document length distribution")
    save(fig, "length_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax, counter, title, color in ((axes[0], raw["counts"], "Before preprocessing", SERIES_COLORS[0]),
                                      (axes[1], processed["counts"], "After preprocessing", SERIES_COLORS[1])):
        top = counter.most_common(15)[::-1]
        ax.barh([t for t, _ in top], [c for _, c in top], color=color)
        ax.set_title(f"Top 15 terms - {title.lower()}")
        ax.set_xlabel("Frequency")
    save(fig, "top_terms_before_after.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for counter, name, color in ((raw["counts"], "Before preprocessing", SERIES_COLORS[0]),
                                 (processed["counts"], "After preprocessing", SERIES_COLORS[1])):
        f = sorted(counter.values(), reverse=True)
        ax.loglog(np.arange(1, len(f) + 1), f, label=name, color=color, linewidth=2)
    ax.set_xlabel("Term rank (log)")
    ax.set_ylabel("Frequency (log)")
    ax.set_title("Term-frequency distribution")
    ax.legend()
    save(fig, "rank_frequency.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(labels))
    actual = [clf["y_true"].count(l) for l in labels]
    pred = [clf["y_pred"].count(l) for l in labels]
    ax.bar(x - 0.2, actual, 0.38, label="Actual", color=SERIES_COLORS[0])
    ax.bar(x + 0.2, pred, 0.38, label="Predicted", color=SERIES_COLORS[1])
    ax.set_xticks(x, short)
    ax.set_ylabel("Test documents")
    ax.set_title("Actual vs predicted class distribution (test set)")
    ax.legend()
    save(fig, "actual_vs_predicted.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    heatmap(axes[0], m["confusion_matrix"], short, short)
    axes[0].set_title("Test split")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    heatmap(axes[1], clf["cv_metrics"]["confusion_matrix"], short, short)
    axes[1].set_title("Pooled 5-fold cross-validation")
    axes[1].set_xlabel("Predicted")
    save(fig, "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for n, key in enumerate(["precision", "recall", "f1"]):
        ax.bar(x + (n - 1) * 0.27, [c[key] for c in m["per_class"]], 0.25, label=key,
               color=SERIES_COLORS[n])
    ax.set_xticks(x, short)
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-class metrics (test set)")
    ax.legend(ncol=3)
    save(fig, "per_class_metrics.png")

    k = clu["kmeans"].k
    coords = clu["coords"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for c in range(k):
        mask = np.array(clu["clusters"]) == c
        axes[0].scatter(coords[mask, 0], coords[mask, 1], s=40, color=SERIES_COLORS[c % 8],
                        edgecolor="white", linewidth=1, label=f"Cluster {c}")
    axes[0].set_title("K-Means clusters (PCA projection)")
    axes[0].legend(fontsize=8)
    for l in labels:
        mask = np.array(clu["labels"]) == l
        axes[1].scatter(coords[mask, 0], coords[mask, 1], s=40, color=CATEGORY_COLORS[l],
                        edgecolor="white", linewidth=1, label=SHORT[l])
    axes[1].set_title("True categories (same projection)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    save(fig, "cluster_scatter.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.4]})
    sizes = np.bincount(clu["clusters"], minlength=k)
    axes[0].bar([f"C{c}" for c in range(k)], sizes, color=[SERIES_COLORS[c % 8] for c in range(k)])
    for c, v in enumerate(sizes):
        axes[0].text(c, v + 0.3, str(v), ha="center")
    axes[0].set_title("Documents per cluster")
    z = [[clu["contingency"].get(c, {}).get(l, 0) for l in labels] for c in range(k)]
    heatmap(axes[1], z, short, [f"Cluster {c}" for c in range(k)])
    axes[1].set_title("Cluster vs true category")
    save(fig, "cluster_sizes_contingency.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(sweep["k"], sweep["purity"], "o-", color=SERIES_COLORS[0], label="Purity", lw=2)
    axes[0].plot(sweep["k"], sweep["rand_index"], "o-", color=SERIES_COLORS[1], label="Rand Index", lw=2)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("K")
    axes[0].legend()
    axes[0].set_title("Purity and Rand Index vs K")
    axes[1].plot(sweep["k"], sweep["sse"], "o-", color=SERIES_COLORS[2], lw=2)
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("SSE")
    axes[1].set_title("SSE vs K")
    save(fig, "k_sweep.png")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    names = [s.split(".")[0] for s in prep_df["setting"]]
    xx = np.arange(len(names))
    for n, (col, lab) in enumerate([("nb_cv_accuracy", "NB accuracy (5-fold CV)"),
                                    ("kmeans_purity", "K-Means purity (mean of 5 seeds)"),
                                    ("kmeans_rand_index", "K-Means Rand Index"),
                                    ("recommendation_p_at_5", "Recommendation P@5")]):
        ax.bar(xx + (n - 1.5) * 0.2, prep_df[col], 0.19, label=lab, color=SERIES_COLORS[n])
    ax.set_xticks(xx, names)
    ax.set_xlabel("Preprocessing setting (see table)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.set_title("Effect of preprocessing settings")
    save(fig, "preprocessing_comparison.png")

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    vals = [m["accuracy"], m["macro_precision"], m["macro_recall"], m["macro_f1"],
            clu["purity"], clu["rand_index"]]
    names = ["Accuracy", "Macro P", "Macro R", "Macro F1", "Purity", "Rand Index"]
    ax.bar(names, vals, color=[SERIES_COLORS[0]] * 4 + [SERIES_COLORS[1]] * 2)
    for n, v in enumerate(vals):
        ax.text(n, v + 0.02, f"{v:.3f}", ha="center")
    ax.set_ylim(0, 1.1)
    ax.set_title("Naive Bayes (blue) vs K-Means (orange)")
    save(fig, "nb_vs_kmeans.png")

    recs = pd.DataFrame(examples["Long short-term memory"]["recommendations"])
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.barh(recs["title"][::-1], recs["similarity"][::-1],
            color=[CATEGORY_COLORS[c] for c in recs["category"][::-1]])
    ax.set_xlabel("Cosine similarity")
    ax.set_title("Top-5 recommendations for 'Long short-term memory'")
    save(fig, "recommendation_example.png")


if __name__ == "__main__":
    main()
