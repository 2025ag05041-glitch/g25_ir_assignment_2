"""Manually implemented evaluation metrics.

Classification: confusion matrix, accuracy, per-class precision / recall / F1,
macro and weighted averages.
Clustering: contingency table, Purity and Rand Index.
"""
from collections import Counter, defaultdict


def confusion_matrix(y_true, y_pred, labels):
    """Rows = actual class, columns = predicted class."""
    position = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for actual, predicted in zip(y_true, y_pred):
        matrix[position[actual]][position[predicted]] += 1
    return matrix


def _safe_divide(a, b):
    return a / b if b else 0.0


def classification_metrics(y_true, y_pred, labels):
    """Accuracy plus per-class and averaged precision, recall and F1.

    precision_c = TP_c / (TP_c + FP_c)
    recall_c    = TP_c / (TP_c + FN_c)
    F1_c        = 2 * P * R / (P + R)
    Macro average = plain mean over classes; weighted average = mean
    weighted by the number of true documents in each class (support).
    """
    matrix = confusion_matrix(y_true, y_pred, labels)
    n = len(y_true)
    correct = sum(matrix[i][i] for i in range(len(labels)))
    per_class = []
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(len(labels))) - tp
        fn = sum(matrix[i]) - tp
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class.append({"class": label, "precision": precision, "recall": recall,
                          "f1": f1, "support": tp + fn})

    def average(key, weighted):
        if weighted:
            total = sum(c["support"] for c in per_class)
            return _safe_divide(sum(c[key] * c["support"] for c in per_class), total)
        return _safe_divide(sum(c[key] for c in per_class), len(per_class))

    return {
        "accuracy": _safe_divide(correct, n),
        "macro_precision": average("precision", False),
        "macro_recall": average("recall", False),
        "macro_f1": average("f1", False),
        "weighted_precision": average("precision", True),
        "weighted_recall": average("recall", True),
        "weighted_f1": average("f1", True),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "labels": list(labels),
    }


def contingency_table(clusters, labels):
    """table[cluster][label] = number of documents in both."""
    table = defaultdict(Counter)
    for cluster, label in zip(clusters, labels):
        table[cluster][label] += 1
    return table


def purity(clusters, labels):
    """Purity = (1/N) * sum over clusters of the size of the majority class."""
    if len(clusters) != len(labels):
        raise ValueError("clusters and labels must have the same length")
    if not clusters:
        return 0.0
    table = contingency_table(clusters, labels)
    return sum(max(counts.values()) for counts in table.values()) / len(clusters)


def _pairs(n):
    return n * (n - 1) // 2


def pair_counts(clusters, labels):
    """TP, FP, FN, TN over all document pairs.

    TP: same cluster and same class      FP: same cluster, different class
    FN: different cluster, same class    TN: different cluster and class
    Computed from the contingency table instead of looping over every pair.
    """
    if len(clusters) != len(labels):
        raise ValueError("clusters and labels must have the same length")
    table = contingency_table(clusters, labels)
    same_cluster = sum(_pairs(sum(c.values())) for c in table.values())
    tp = sum(_pairs(n) for c in table.values() for n in c.values())
    same_class = sum(_pairs(n) for n in Counter(labels).values())
    fp = same_cluster - tp
    fn = same_class - tp
    tn = _pairs(len(labels)) - tp - fp - fn
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def rand_index(clusters, labels):
    """RI = (TP + TN) / (TP + FP + FN + TN)."""
    counts = pair_counts(clusters, labels)
    total = sum(counts.values())
    if total == 0:
        return 1.0
    return (counts["TP"] + counts["TN"]) / total


def rand_index_bruteforce(clusters, labels):
    """Direct O(N^2) version used in the tests to check rand_index()."""
    agree = total = 0
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            same_cluster = clusters[i] == clusters[j]
            same_class = labels[i] == labels[j]
            agree += same_cluster == same_class
            total += 1
    return agree / total if total else 1.0
