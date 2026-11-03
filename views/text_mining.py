import numpy as np
import pandas as pd
import streamlit as st

from src import visualization as viz
from src.app_state import (get_classification, get_clustering, get_km_params, get_model,
                           get_nb_params, preprocess_query)

st.title("Text Mining & Corpus Analytics")
model = get_model()
docs = model.docs
nb_tab, km_tab = st.tabs(["Naive Bayes classification", "K-Means clustering"])

# ------------------------------------------------------------------ Naive Bayes
with nb_tab:
    st.markdown(r"""
Multinomial Naive Bayes implemented from scratch (`src/classification.py`), trained on the term counts
produced by the current preprocessing settings. Vocabulary, priors and likelihoods use **training documents only**.
""")
    st.latex(r"P(c)=\frac{N_c}{N}\qquad P(t\mid c)=\frac{count(t,c)+\alpha}{\sum_{t'}count(t',c)+\alpha|V|}"
             r"\qquad \hat c=\arg\max_c\Big[\log P(c)+\sum_{t\in d} tf(t,d)\log P(t\mid c)\Big]")
    p = get_nb_params()
    with st.form("nb_form"):
        c1, c2, c3, c4 = st.columns(4)
        test_size = c1.slider("Test set size", 0.2, 0.5, p["test_size"], 0.05)
        seed = c2.number_input("Random seed", 0, 9999, p["seed"])
        alpha = c3.number_input("Laplace smoothing α", 0.01, 5.0, p["alpha"], 0.1)
        group = c4.checkbox("Keep duplicate groups in one split", p["group_duplicates"])
        if st.form_submit_button("Train and evaluate", type="primary"):
            st.session_state["nb_params"] = {"test_size": test_size, "seed": int(seed),
                                             "alpha": float(alpha), "group_duplicates": group}
            st.rerun()

    result = get_classification()
    m = result["metrics"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Training documents", len(result["train_idx"]))
    c2.metric("Test documents", len(result["test_idx"]))
    c3.metric("Accuracy", f"{m['accuracy']:.3f}")
    c4.metric("Macro precision", f"{m['macro_precision']:.3f}")
    c5.metric("Macro recall", f"{m['macro_recall']:.3f}")
    c6.metric("Macro F1", f"{m['macro_f1']:.3f}")

    st.markdown("**Actual vs predicted labels for the test documents**")
    predictions = pd.DataFrame({
        "doc_id": docs["doc_id"].iloc[result["test_idx"]].values,
        "title": docs["title"].iloc[result["test_idx"]].values,
        "actual": result["y_true"],
        "predicted": result["y_pred"],
    })
    predictions["correct"] = np.where(predictions["actual"] == predictions["predicted"], "yes", "NO")
    st.dataframe(predictions, hide_index=True, width="stretch")

    st.markdown("**Per-class metrics**")
    per_class = pd.DataFrame(m["per_class"]).round(3)
    st.dataframe(per_class, hide_index=True, width="stretch")
    st.caption("Macro average = simple mean over classes (every class counts equally). "
               "Weighted average = mean weighted by support. "
               f"Weighted F1 = {m['weighted_f1']:.3f}.")

    cv = result.get("cv_metrics")
    if cv:
        st.markdown(f"**5-fold cross-validation** (more stable than a single small test set): "
                    f"accuracy = {cv['accuracy']:.3f}, macro F1 = {cv['macro_f1']:.3f}; "
                    f"fold accuracies = {', '.join(f'{a:.2f}' for a in cv['fold_accuracy'])}")

    st.markdown("**Most indicative terms per class** (log P(t|c) minus mean log P(t|other classes))")
    top = result["model"].top_terms(10)
    st.dataframe(pd.DataFrame({c: [t for t, _ in terms] for c, terms in top.items()}),
                 hide_index=True, width="stretch")

    st.markdown("**Classify new text**")
    text = st.text_area("Paste a paragraph to classify",
                        "Central banks raised interest rates and bond yields climbed as investors "
                        "moved money from equities into money market funds.")
    if text.strip():
        terms = preprocess_query(text)
        known = [t for t in terms if t in result["model"].vocabulary]
        if not known:
            st.warning("None of the terms are in the training vocabulary; the prediction would only "
                       "use the class priors.")
        post = result["model"].posterior(terms)
        st.write(f"Predicted category: **{max(post, key=post.get)}** "
                 f"({len(known)} of {len(terms)} terms known to the model)")
        st.dataframe(pd.DataFrame(sorted(post.items(), key=lambda x: -x[1]),
                                  columns=["category", "posterior probability"]).round(4),
                     hide_index=True, width="stretch")

# ------------------------------------------------------------------ K-Means
with km_tab:
    st.markdown("""
K-Means implemented from scratch (`src/clustering.py`) on the L2-normalised TF-IDF vectors of the
labelled documents. Distance is squared Euclidean distance; for unit-length vectors this ranks
documents the same way as cosine similarity.
""")
    p = get_km_params()
    with st.form("km_form"):
        c1, c2, c3, c4, c5 = st.columns(5)
        k = c1.slider("Number of clusters K", 2, 12, p["k"])
        init = c2.selectbox("Initialisation", ["k-means++", "random"],
                            index=["k-means++", "random"].index(p["init"]))
        n_init = c3.number_input("Number of runs (keep lowest SSE)", 1, 50, p["n_init"])
        max_iter = c4.number_input("Max iterations", 1, 500, p["max_iter"])
        seed = c5.number_input("Random seed", 0, 9999, p["seed"])
        if st.form_submit_button("Run K-Means", type="primary"):
            st.session_state["km_params"] = {"k": k, "init": init, "n_init": int(n_init),
                                             "max_iter": int(max_iter), "seed": int(seed)}
            st.rerun()

    res = get_clustering()
    km = res["kmeans"]
    idx = res["indices"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("K", km.k)
    c2.metric("Iterations (best run)", km.n_iter_)
    c3.metric("Converged", "Yes" if km.converged_ else "No (max iterations)")
    c4.metric("Purity", f"{res['purity']:.3f}")
    c5.metric("Rand Index", f"{res['rand_index']:.3f}")

    st.markdown("**Step 1 - Centroid initialisation** "
                f"({km.init}, run {km.best_run + 1} of {km.n_init} had the lowest SSE)")
    seeds = pd.DataFrame({
        "cluster": range(km.k),
        "seed doc_id": [docs["doc_id"].iloc[idx[s]] for s in km.initial_seeds_],
        "seed document": [docs["title"].iloc[idx[s]] for s in km.initial_seeds_],
        "seed category": [docs["category"].iloc[idx[s]] for s in km.initial_seeds_],
    })
    st.dataframe(seeds, hide_index=True, width="stretch")

    st.markdown("**Steps 2-4 - Assignment, centroid update and stopping criterion**")
    st.markdown("Each iteration assigns every document to its nearest centroid, then recomputes each "
                "centroid as the mean of its documents. The loop stops when no document changes "
                "cluster or the largest centroid shift is below 1e-6, or after the maximum number "
                "of iterations. Empty clusters are re-seeded with the worst-fitting document.")
    history = pd.DataFrame(km.history_)
    history["sizes"] = history["sizes"].astype(str)
    st.dataframe(history.round(5), hide_index=True, width="stretch")
    runs = pd.DataFrame([{"run": r + 1, "SSE": round(run["sse"], 4), "iterations": run["n_iter"],
                          "converged": run["converged"]} for r, run in enumerate(km.runs)])
    with st.expander("All runs"):
        st.dataframe(runs, hide_index=True, width="stretch")

    left, right = st.columns(2)
    left.plotly_chart(viz.sse_by_iteration(km.history_))
    right.plotly_chart(viz.cluster_sizes(np.array(res["clusters"]), km.k))

    st.markdown("**Cluster descriptions** (highest-weighted centroid terms and majority category)")
    descr = []
    for c in range(km.k):
        counts = res["contingency"].get(c, {})
        size = sum(counts.values())
        majority = max(counts, key=counts.get) if counts else "-"
        descr.append({"cluster": c, "size": size, "majority category": majority,
                      "majority share": round(counts.get(majority, 0) / size, 3) if size else 0,
                      "top terms": ", ".join(res["top_terms"][c])})
    st.dataframe(pd.DataFrame(descr), hide_index=True, width="stretch")

    st.markdown("**Cluster membership**")
    members = pd.DataFrame({"doc_id": docs["doc_id"].iloc[idx].values,
                            "title": docs["title"].iloc[idx].values,
                            "true label": res["labels"], "assigned cluster": res["clusters"]})
    cluster_filter = st.multiselect("Show clusters", list(range(km.k)))
    if cluster_filter:
        members = members[members["assigned cluster"].isin(cluster_filter)]
    st.dataframe(members.sort_values(["assigned cluster", "true label"]), hide_index=True,
                 width="stretch")
