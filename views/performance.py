import json

import numpy as np
import pandas as pd
import streamlit as st

from config import CATEGORY_COLORS, RESULTS_DIR
from src import visualization as viz
from src.app_state import get_classification, get_clustering, get_corpus, get_km_params, get_model
from src.experiments import compare_preprocessing, duplicate_experiment, k_sweep

st.title("Performance Analysis")
st.caption("Results use the settings chosen on the Preprocessing and Text Mining pages.")

model = get_model()
docs = model.docs
clf = get_classification()
clu = get_clustering()
m = clf["metrics"]
labels = m["labels"]

cls_tab, clu_tab, cmp_tab, exp_tab = st.tabs(
    ["Classification", "Clustering", "Comparison", "Experiments"])

with cls_tab:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("Macro precision", f"{m['macro_precision']:.3f}")
    c3.metric("Macro recall", f"{m['macro_recall']:.3f}")
    c4.metric("Macro F1", f"{m['macro_f1']:.3f}")
    left, right = st.columns(2)
    left.plotly_chart(viz.actual_vs_predicted(clf["y_true"], clf["y_pred"], labels))
    right.plotly_chart(viz.confusion_heatmap(m["confusion_matrix"], labels,
                                             "Confusion matrix (test set)"))
    st.plotly_chart(viz.per_class_metrics(m["per_class"]))
    cv = clf.get("cv_metrics")
    if cv:
        st.markdown(f"**5-fold cross-validation over all {len(model.eval_indices)} labelled documents**: "
                    f"accuracy {cv['accuracy']:.3f}, macro F1 {cv['macro_f1']:.3f}")
        st.plotly_chart(viz.confusion_heatmap(cv["confusion_matrix"], labels,
                                              "Confusion matrix (pooled 5-fold cross-validation)"))

with clu_tab:
    km = clu["kmeans"]
    idx = clu["indices"]
    c1, c2, c3 = st.columns(3)
    c1.metric("K", km.k)
    c2.metric("Purity", f"{clu['purity']:.3f}")
    c3.metric("Rand Index", f"{clu['rand_index']:.3f}")
    color_by = st.radio("Colour points by", ["Assigned cluster", "True category"], horizontal=True)
    titles = docs["title"].iloc[idx].tolist()
    if color_by == "Assigned cluster":
        fig = viz.cluster_scatter(clu["coords"], [f"Cluster {c}" for c in clu["clusters"]], titles,
                                  "Cluster", None)
    else:
        fig = viz.cluster_scatter(clu["coords"], clu["labels"], titles, "Category", CATEGORY_COLORS)
    st.plotly_chart(fig)
    var = clu["explained_variance"]
    st.caption(f"PCA is used only to draw this plot (PC1 + PC2 explain {sum(var):.1%} of the variance). "
               "K-Means runs on the full TF-IDF vectors, so clusters can overlap in 2D.")
    left, right = st.columns(2)
    left.plotly_chart(viz.cluster_sizes(np.array(clu["clusters"]), km.k))
    right.plotly_chart(viz.contingency_heatmap(clu["contingency"], km.k, labels))
    st.markdown("**Cluster membership table**")
    st.dataframe(pd.DataFrame({"doc_id": docs["doc_id"].iloc[idx].values, "title": titles,
                               "true label": clu["labels"], "cluster": clu["clusters"]})
                 .sort_values(["cluster", "true label"]), hide_index=True, width="stretch")
    pc = clu["pair_counts"]
    st.markdown(f"""
**What the metrics mean**
- **Purity** = (1/N) Σ_k max_j |cluster_k ∩ class_j|: each cluster is credited with its most common true
  category. {clu['purity']:.3f} means {clu['purity']:.1%} of documents are in a cluster whose majority category is their own.
- **Rand Index** = (TP + TN) / all pairs: share of the {sum(pc.values()):,} document pairs on which the
  clustering and the true labels agree. TP = {pc['TP']:,} (same cluster, same class),
  TN = {pc['TN']:,} (different cluster, different class), FP = {pc['FP']:,}, FN = {pc['FN']:,}.
- Purity rises with K (with K = N it is 1.0) and RI is high when most pairs are in different classes, so
  both should be read together and for a fixed K.
""")
    if st.checkbox("Show Purity / Rand Index / SSE for K = 2..10"):
        sweep = k_sweep(model, seed=get_km_params()["seed"])
        left, right = st.columns(2)
        left.plotly_chart(viz.k_sweep_chart(sweep))
        right.plotly_chart(viz.k_sweep_sse(sweep))
        st.dataframe(sweep.round(4), hide_index=True, width="stretch")

with cmp_tab:
    cv = clf.get("cv_metrics", m)
    table = pd.DataFrame([
        {"method": "Naive Bayes (supervised)", "metric": "Accuracy (test split)", "value": m["accuracy"]},
        {"method": "Naive Bayes (supervised)", "metric": "Macro F1 (test split)", "value": m["macro_f1"]},
        {"method": "Naive Bayes (supervised)", "metric": "Accuracy (5-fold CV)", "value": cv["accuracy"]},
        {"method": "Naive Bayes (supervised)", "metric": "Macro F1 (5-fold CV)", "value": cv["macro_f1"]},
        {"method": f"K-Means, K={clu['kmeans'].k} (unsupervised)", "metric": "Purity", "value": clu["purity"]},
        {"method": f"K-Means, K={clu['kmeans'].k} (unsupervised)", "metric": "Rand Index",
         "value": clu["rand_index"]},
    ])
    st.dataframe(table.round(4), hide_index=True, width="stretch")
    st.plotly_chart(viz.comparison_chart(m, clu["purity"], clu["rand_index"]))
    worst = min(m["per_class"], key=lambda c: c["f1"])
    st.markdown(f"""
**Interpretation**
- Naive Bayes learns from labelled examples and directly predicts our five categories. On the test set it
  reaches an accuracy of **{m['accuracy']:.3f}** (macro F1 **{m['macro_f1']:.3f}**). The weakest class is
  *{worst['class']}* (F1 {worst['f1']:.3f}).
- K-Means never sees the labels. It groups documents by vocabulary overlap, and we only compare its groups
  to the categories afterwards. A purity of **{clu['purity']:.3f}** shows that the discovered groups partly follow our
  categories, but some clusters mix topics or split a large category.
- The numbers are **not directly comparable**. Accuracy/F1 measure agreement with the labels for each document;
  Purity/RI measure how well an unlabeled grouping matches the labels. Supervision gives
  Naive Bayes a clear advantage, while K-Means is useful for discovering structure when no labels exist.
- With only {len(clf['test_idx'])} test documents, one document changes accuracy by about
  {1 / max(len(clf['test_idx']), 1):.1%}; cross-validation gives a more stable estimate.
""")

with exp_tab:
    st.markdown("These experiments produce the evidence for the inference questions in the report. "
                "Saved results come from `python scripts/run_experiments.py` using the default settings. "
                "You can also run them again here (takes about 30-60 seconds).")
    saved_prep = RESULTS_DIR / "preprocessing_comparison.csv"
    saved_dup = RESULTS_DIR / "duplicate_experiment.json"

    st.subheader("1. Effect of preprocessing choices")
    if st.button("Run preprocessing comparison now"):
        bar = st.progress(0.0)
        st.session_state["prep_comparison"] = compare_preprocessing(
            get_corpus(), progress=lambda f, name: bar.progress(f, text=name))
    prep_df = st.session_state.get("prep_comparison")
    if prep_df is None and saved_prep.exists():
        prep_df = pd.read_csv(saved_prep)
        st.caption("Showing saved results.")
    if prep_df is not None:
        st.dataframe(prep_df.round(3), hide_index=True, width="stretch")
        st.caption("NB accuracy/F1 use one fixed 70/30 split (seed 42); CV columns use 5-fold "
                   "cross-validation. K-Means values are the mean over 5 seeds (K = 5).")

    st.subheader("2. Effect of duplicate documents (simulation)")
    st.markdown("Our real corpus has no duplicates, so this experiment adds in memory one **exact copy** and "
                "one **edited copy** (10% of words deleted) of 10 labelled documents, marked `[SIMULATED]`. "
                "The dataset files are not changed.")
    if st.button("Run duplicate experiment now"):
        with st.spinner("Running..."):
            st.session_state["dup_experiment"] = duplicate_experiment(get_corpus())
    dup = st.session_state.get("dup_experiment")
    if dup is None and saved_dup.exists():
        dup = json.loads(saved_dup.read_text())
        st.caption("Showing saved results.")
    if dup:
        d, c, r, k = dup["detection"], dup["classification"], dup["recommendation"], dup["clustering"]
        st.markdown(f"""
| Aspect | Without duplicates | With duplicates (no handling) | With duplicates (our handling) |
|---|---|---|---|
| Detection | - | - | {d['detected_injected_pairs']}/{d['injected_pairs']} injected pairs found, {d['false_positive_pairs']} false positives |
| NB accuracy (mean of {c['splits']} splits) | {c['original_corpus']:.3f} | {c['with_duplicates_naive_split']:.3f} (avg {c['avg_leaked_test_docs_naive_split']:.1f} test docs have a copy in training) | {c['with_duplicates_group_split']:.3f} (group-aware split) |
| Top-5 slots used by copies | - | {r['share_of_top5_slots_taken_by_copies_unfiltered']:.0%} | {r['share_of_top5_slots_taken_by_copies_filtered']:.0%} (near-duplicates hidden) |
| K-Means purity / RI | {k['purity_original']:.3f} / {k['rand_index_original']:.3f} | {k['purity_with_duplicates']:.3f} / {k['rand_index_with_duplicates']:.3f} | {k['purity_after_dedup']:.3f} / {k['rand_index_after_dedup']:.3f} (copies removed) |
""")
        st.caption(f"{k['copies_in_same_cluster_as_source']} of {k['copies_total']} copies were placed in the "
                   "same cluster as their source document.")
