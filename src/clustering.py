"""Manual K-Means clustering on the L2-normalised TF-IDF document vectors.

Distance: squared Euclidean distance. Because every document vector has unit
length, ||x - c||^2 = 1 + ||c||^2 - 2 x.c, so documents with a high cosine
similarity to a centroid are also close to it in Euclidean distance.

Steps
  1. Initialisation: k-means++ (spread-out seeds) or k random documents.
  2. Assignment:     each document goes to its nearest centroid.
  3. Update:         each centroid becomes the mean of its documents.
                     An empty cluster is re-seeded with the document that is
                     currently farthest from its own centroid.
  4. Stopping:       no document changed cluster, or the largest centroid
                     movement is below `tol`, or `max_iter` is reached.
The whole procedure is repeated `n_init` times with different seeds and the
run with the lowest SSE (within-cluster sum of squares) is kept.
NumPy is used only for the vector arithmetic.
"""
import numpy as np


def squared_distances(X, centroids):
    """Matrix of squared Euclidean distances, shape (n_docs, k)."""
    x_sq = (X * X).sum(axis=1)[:, None]
    c_sq = (centroids * centroids).sum(axis=1)[None, :]
    return np.maximum(x_sq + c_sq - 2 * X @ centroids.T, 0.0)


class KMeans:
    def __init__(self, k=5, init="k-means++", max_iter=100, tol=1e-6,
                 n_init=10, seed=42):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.seed = seed

    def _initial_seeds(self, X, rng):
        n = X.shape[0]
        if self.init == "random":
            return list(rng.choice(n, self.k, replace=False))
        seeds = [int(rng.randint(n))]
        for _ in range(1, self.k):
            dist = squared_distances(X, X[seeds]).min(axis=1)
            if dist.sum() == 0:
                remaining = [i for i in range(n) if i not in seeds]
                seeds.append(int(rng.choice(remaining)))
            else:
                seeds.append(int(rng.choice(n, p=dist / dist.sum())))
        return seeds

    def _single_run(self, X, rng):
        seeds = self._initial_seeds(X, rng)
        centroids = X[seeds].copy()
        labels = np.full(X.shape[0], -1)
        history = []
        converged = False
        reseeded = 0
        for iteration in range(1, self.max_iter + 1):
            dist = squared_distances(X, centroids)
            new_labels = dist.argmin(axis=1)
            changed = int((new_labels != labels).sum())
            labels = new_labels

            new_centroids = np.zeros_like(centroids)
            for c in range(self.k):
                members = labels == c
                if members.any():
                    new_centroids[c] = X[members].mean(axis=0)
                else:
                    # empty cluster: move it to the worst-fitting document
                    own = dist[np.arange(len(labels)), labels]
                    far = int(own.argmax())
                    new_centroids[c] = X[far]
                    labels[far] = c
                    reseeded += 1
            shift = float(np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max())
            centroids = new_centroids
            sse = float(squared_distances(X, centroids)[np.arange(len(labels)), labels].sum())
            history.append({"iteration": iteration, "sse": sse, "changed": changed,
                            "max_centroid_shift": shift,
                            "sizes": np.bincount(labels, minlength=self.k).tolist()})
            if changed == 0 or shift < self.tol:
                converged = True
                break
        return {"labels": labels, "centroids": centroids, "sse": history[-1]["sse"],
                "n_iter": len(history), "converged": converged, "history": history,
                "initial_seeds": [int(s) for s in seeds], "reseeded": reseeded}

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.shape[0] < self.k:
            raise ValueError(f"cannot form {self.k} clusters from {X.shape[0]} documents")
        rng = np.random.RandomState(self.seed)
        self.runs = [self._single_run(X, rng) for _ in range(self.n_init)]
        best = min(range(len(self.runs)), key=lambda r: self.runs[r]["sse"])
        self.best_run = best
        result = self.runs[best]
        self.labels_ = result["labels"]
        self.centroids_ = result["centroids"]
        self.sse_ = result["sse"]
        self.n_iter_ = result["n_iter"]
        self.converged_ = result["converged"]
        self.history_ = result["history"]
        self.initial_seeds_ = result["initial_seeds"]
        return self


def centroid_top_terms(centroids, terms, n=8):
    """Highest-weighted terms of each centroid (used to describe clusters)."""
    result = []
    for row in centroids:
        top = np.argsort(-row)[:n]
        result.append([terms[i] for i in top if row[i] > 0])
    return result


def project_2d(X):
    """PCA to two dimensions via SVD. Used only to draw the scatter plot;
    the clustering itself runs on the full TF-IDF vectors."""
    X = np.asarray(X, dtype=float)
    centered = X - X.mean(axis=0)
    if centered.shape[0] < 2:
        return np.zeros((centered.shape[0], 2)), [0.0, 0.0]
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :2] * s[:2]
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])
    variance = (s ** 2) / max((s ** 2).sum(), 1e-12)
    return coords, variance[:2].tolist()
