"""Plotly chart helpers used by the Streamlit pages."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import CATEGORY_COLORS, SERIES_COLORS

BASE_LAYOUT = dict(template="plotly_white", font=dict(size=13),
                   margin=dict(l=10, r=10, t=50, b=10), bargap=0.25)


def _finish(fig, title, height=380):
    fig.update_layout(title=title, height=height, **BASE_LAYOUT)
    return fig


def category_distribution(df):
    counts = df["category"].value_counts().rename_axis("category").reset_index(name="documents")
    fig = px.bar(counts, x="documents", y="category", orientation="h", color="category",
                 color_discrete_map=CATEGORY_COLORS, text="documents")
    fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Number of documents")
    fig.update_yaxes(categoryorder="total ascending")
    return _finish(fig, "Documents per category", 320)


def length_histogram(df):
    fig = px.histogram(df, x="word_count", nbins=25, color_discrete_sequence=[SERIES_COLORS[0]])
    fig.update_layout(xaxis_title="Words per document", yaxis_title="Documents")
    return _finish(fig, "Document length distribution", 320)


def length_by_category(df):
    fig = px.box(df, x="word_count", y="category", color="category",
                 color_discrete_map=CATEGORY_COLORS, points="all", hover_name="title")
    fig.update_layout(showlegend=False, xaxis_title="Words per document", yaxis_title="")
    return _finish(fig, "Document length by category", 340)


def top_terms_bar(counter, n=20, title="Most frequent terms", color=SERIES_COLORS[0]):
    items = counter.most_common(n)
    data = pd.DataFrame(items, columns=["term", "frequency"])
    fig = px.bar(data, x="frequency", y="term", orientation="h",
                 color_discrete_sequence=[color])
    fig.update_yaxes(categoryorder="total ascending", title="")
    fig.update_xaxes(title="Frequency")
    return _finish(fig, title, 20 * n + 120)


def rank_frequency(before, after):
    """Zipf-style rank/frequency curves before and after preprocessing."""
    fig = go.Figure()
    for name, counter, color in (("Before preprocessing", before, SERIES_COLORS[0]),
                                 ("After preprocessing", after, SERIES_COLORS[1])):
        freqs = sorted(counter.values(), reverse=True)
        fig.add_trace(go.Scatter(x=np.arange(1, len(freqs) + 1), y=freqs, mode="lines",
                                 name=name, line=dict(width=2, color=color)))
    fig.update_xaxes(type="log", title="Term rank (log scale)")
    fig.update_yaxes(type="log", title="Frequency (log scale)")
    return _finish(fig, "Term-frequency distribution (rank vs frequency)", 360)


def df_histogram(df_values):
    fig = px.histogram(x=df_values, nbins=40, color_discrete_sequence=[SERIES_COLORS[2]])
    fig.update_layout(xaxis_title="Document frequency of term (df)", yaxis_title="Number of terms")
    return _finish(fig, "Document frequency distribution of TF-IDF vocabulary", 320)


def similarity_heatmap(sim, labels):
    fig = go.Figure(go.Heatmap(z=sim, x=labels, y=labels, colorscale="Blues", zmin=0, zmax=1,
                               colorbar=dict(title="cosine")))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")
    return _finish(fig, "Cosine similarity between all documents (sorted by category)", 520)


def actual_vs_predicted(y_true, y_pred, labels):
    data = pd.DataFrame({
        "category": labels * 2,
        "count": [y_true.count(l) for l in labels] + [y_pred.count(l) for l in labels],
        "series": ["Actual"] * len(labels) + ["Predicted"] * len(labels),
    })
    fig = px.bar(data, x="category", y="count", color="series", barmode="group",
                 color_discrete_sequence=SERIES_COLORS[:2], text="count")
    fig.update_layout(xaxis_title="", yaxis_title="Test documents", legend_title="")
    return _finish(fig, "Actual vs predicted class distribution (test set)")


def confusion_heatmap(matrix, labels, title="Confusion matrix"):
    z = np.array(matrix)
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=labels, colorscale="Blues", showscale=False,
                               text=z, texttemplate="%{text}", textfont=dict(size=15)))
    fig.update_xaxes(title="Predicted label", side="bottom")
    fig.update_yaxes(title="Actual label", autorange="reversed")
    return _finish(fig, title, 420)


def per_class_metrics(per_class):
    data = pd.DataFrame(per_class).melt(id_vars=["class"], value_vars=["precision", "recall", "f1"],
                                        var_name="metric", value_name="score")
    fig = px.bar(data, x="class", y="score", color="metric", barmode="group",
                 color_discrete_sequence=SERIES_COLORS[:3])
    fig.update_layout(yaxis_range=[0, 1.05], xaxis_title="", legend_title="")
    return _finish(fig, "Precision, recall and F1 per class")


def cluster_scatter(coords, color_values, hover, color_title, color_map=None):
    data = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1],
                         color_title: color_values, "title": hover})
    order = sorted(set(color_values), key=lambda v: (len(str(v)), str(v)))
    if color_map is None:
        # cluster i always gets colour i, the same as in the cluster-size chart
        color_map = {v: SERIES_COLORS[n % len(SERIES_COLORS)] for n, v in enumerate(order)}
    fig = px.scatter(data, x="PC1", y="PC2", color=color_title, hover_name="title",
                     color_discrete_map=color_map, category_orders={color_title: order})
    fig.update_traces(marker=dict(size=11, line=dict(width=1.5, color="white")))
    return _finish(fig, f"Documents in 2D (PCA projection), coloured by {color_title.lower()}", 460)


def cluster_sizes(clusters, k):
    sizes = np.bincount(clusters, minlength=k)
    data = pd.DataFrame({"cluster": [f"Cluster {i}" for i in range(k)], "documents": sizes})
    fig = px.bar(data, x="cluster", y="documents", text="documents",
                 color="cluster", color_discrete_sequence=SERIES_COLORS)
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Documents")
    return _finish(fig, "Number of documents per cluster", 340)


def contingency_heatmap(table, k, labels):
    z = [[table.get(c, {}).get(l, 0) for l in labels] for c in range(k)]
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=[f"Cluster {c}" for c in range(k)],
                               colorscale="Blues", showscale=False, text=z,
                               texttemplate="%{text}", textfont=dict(size=14)))
    fig.update_yaxes(autorange="reversed")
    return _finish(fig, "Cluster vs true category (document counts)", 380)


def sse_by_iteration(history):
    data = pd.DataFrame(history)
    fig = px.line(data, x="iteration", y="sse", markers=True,
                  color_discrete_sequence=[SERIES_COLORS[0]])
    fig.update_traces(line=dict(width=2), marker=dict(size=9))
    fig.update_layout(xaxis_title="Iteration", yaxis_title="SSE", xaxis=dict(dtick=1))
    return _finish(fig, "Within-cluster SSE per iteration (best run)", 320)


def k_sweep_chart(sweep):
    data = sweep.melt(id_vars="k", value_vars=["purity", "rand_index"],
                      var_name="metric", value_name="score")
    fig = px.line(data, x="k", y="score", color="metric", markers=True,
                  color_discrete_sequence=SERIES_COLORS[:2])
    fig.update_traces(line=dict(width=2), marker=dict(size=9))
    fig.update_layout(yaxis_range=[0, 1.05], xaxis=dict(dtick=1), legend_title="")
    return _finish(fig, "Purity and Rand Index for different K", 340)


def k_sweep_sse(sweep):
    fig = px.line(sweep, x="k", y="sse", markers=True, color_discrete_sequence=[SERIES_COLORS[2]])
    fig.update_traces(line=dict(width=2), marker=dict(size=9))
    fig.update_layout(xaxis=dict(dtick=1), yaxis_title="SSE")
    return _finish(fig, "SSE for different K (elbow view)", 340)


def recommendation_scores(recs):
    data = recs.assign(label=recs["doc_id"] + " " + recs["title"])
    fig = px.bar(data, x="similarity", y="label", orientation="h", color="category",
                 color_discrete_map=CATEGORY_COLORS, text=data["similarity"].round(3),
                 category_orders={"label": data["label"].tolist()})
    fig.update_yaxes(title="")
    fig.update_layout(xaxis_title="Cosine similarity", legend_title="")
    return _finish(fig, "Top-K recommendations by similarity", 60 * len(recs) + 140)


def comparison_chart(nb_metrics, purity, rand_index):
    data = pd.DataFrame({
        "metric": ["Accuracy", "Macro precision", "Macro recall", "Macro F1", "Purity", "Rand Index"],
        "score": [nb_metrics["accuracy"], nb_metrics["macro_precision"],
                  nb_metrics["macro_recall"], nb_metrics["macro_f1"], purity, rand_index],
        "method": ["Naive Bayes (supervised)"] * 4 + ["K-Means (unsupervised)"] * 2,
    })
    fig = px.bar(data, x="metric", y="score", color="method", text=data["score"].round(3),
                 color_discrete_sequence=SERIES_COLORS[:2])
    fig.update_layout(yaxis_range=[0, 1.1], xaxis_title="", legend_title="")
    return _finish(fig, "Naive Bayes metrics vs K-Means metrics")
