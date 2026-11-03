"""Manual Multinomial Naive Bayes with Laplace smoothing, plus data splitting.

    P(c)       = N_c / N                                  (class prior)
    P(t | c)   = (count(t, c) + alpha) / (sum_t' count(t', c) + alpha * |V|)
    score(c|d) = log P(c) + sum_{t in d} tf(t, d) * log P(t | c)
    predicted  = argmax_c score(c | d)

The vocabulary V and all counts come from the training documents only, so
the test documents never influence the model. Terms in a test document that
were never seen in training are ignored.
"""
import math
from collections import Counter, defaultdict

import numpy as np

from src.evaluation import classification_metrics


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha

    def fit(self, docs_terms, labels):
        if not docs_terms:
            raise ValueError("no training documents")
        self.classes = sorted(set(labels))
        class_docs = Counter(labels)
        n_docs = len(labels)
        self.log_prior = {c: math.log(class_docs[c] / n_docs) for c in self.classes}

        self.term_counts = {c: Counter() for c in self.classes}
        for terms, label in zip(docs_terms, labels):
            self.term_counts[label].update(terms)
        self.vocabulary = set()
        for counts in self.term_counts.values():
            self.vocabulary.update(counts)

        v = len(self.vocabulary)
        self.log_likelihood = {}
        self.log_unseen = {}
        for c in self.classes:
            total = sum(self.term_counts[c].values())
            denominator = total + self.alpha * v
            self.log_likelihood[c] = {
                t: math.log((n + self.alpha) / denominator)
                for t, n in self.term_counts[c].items()}
            # a vocabulary term never seen in class c only gets alpha
            self.log_unseen[c] = math.log(self.alpha / denominator)
        return self

    def class_scores(self, terms):
        counts = Counter(t for t in terms if t in self.vocabulary)
        scores = {}
        for c in self.classes:
            likelihood = self.log_likelihood[c]
            unseen = self.log_unseen[c]
            scores[c] = self.log_prior[c] + sum(
                n * likelihood.get(t, unseen) for t, n in counts.items())
        return scores

    def predict_one(self, terms):
        scores = self.class_scores(terms)
        return max(self.classes, key=lambda c: (scores[c], c))

    def predict(self, docs_terms):
        return [self.predict_one(terms) for terms in docs_terms]

    def posterior(self, terms):
        """Normalised class probabilities (softmax of the log scores)."""
        scores = self.class_scores(terms)
        top = max(scores.values())
        exp = {c: math.exp(s - top) for c, s in scores.items()}
        total = sum(exp.values())
        return {c: v / total for c, v in exp.items()}

    def top_terms(self, n=10):
        """Most class-indicative terms: log P(t|c) minus the average
        log P(t|c') over the other classes."""
        result = {}
        for c in self.classes:
            others = [o for o in self.classes if o != c]
            ranked = []
            for t in self.term_counts[c]:
                if self.term_counts[c][t] < 3:
                    continue
                own = self.log_likelihood[c][t]
                rest = sum(self.log_likelihood[o].get(t, self.log_unseen[o])
                           for o in others) / max(len(others), 1)
                ranked.append((t, own - rest))
            ranked.sort(key=lambda x: -x[1])
            result[c] = ranked[:n]
        return result


def stratified_split(labels, test_size=0.3, seed=42, groups=None):
    """Stratified train/test split that keeps each group on one side.

    groups[i] is a group id for document i; documents that are duplicates of
    each other share a group, so a copy can never be in training while its
    original is in testing (this would leak test information).
    Returns (train_indices, test_indices).
    """
    if groups is None:
        groups = list(range(len(labels)))
    group_members = defaultdict(list)
    for i, g in enumerate(groups):
        group_members[g].append(i)
    # each group is stratified by the majority label of its members
    by_class = defaultdict(list)
    for g, members in group_members.items():
        label = Counter(labels[i] for i in members).most_common(1)[0][0]
        by_class[label].append(g)

    rng = np.random.RandomState(seed)
    train, test = [], []
    for label in sorted(by_class):
        # order groups by their first document so the split does not depend on
        # how the group ids are numbered
        class_groups = sorted(by_class[label], key=lambda g: group_members[g][0])
        rng.shuffle(class_groups)
        n_test = int(round(test_size * len(class_groups)))
        if len(class_groups) >= 2:
            n_test = min(max(n_test, 1), len(class_groups) - 1)
        else:
            n_test = 0
        for k, g in enumerate(class_groups):
            (test if k < n_test else train).extend(group_members[g])
    return sorted(train), sorted(test)


def random_split(n_docs, test_size=0.3, seed=42):
    """Plain random split (no stratification, no grouping)."""
    order = np.random.RandomState(seed).permutation(n_docs)
    n_test = int(round(test_size * n_docs))
    return sorted(order[n_test:].tolist()), sorted(order[:n_test].tolist())


def train_and_evaluate(docs_terms, labels, train_idx, test_idx, alpha=1.0):
    model = MultinomialNaiveBayes(alpha).fit([docs_terms[i] for i in train_idx],
                                             [labels[i] for i in train_idx])
    y_true = [labels[i] for i in test_idx]
    y_pred = model.predict([docs_terms[i] for i in test_idx])
    class_labels = sorted(set(labels))
    metrics = classification_metrics(y_true, y_pred, class_labels)
    return model, y_true, y_pred, metrics


def stratified_folds(labels, k=5, seed=42, groups=None):
    """Assign groups to k folds, class by class, in round-robin order."""
    if groups is None:
        groups = list(range(len(labels)))
    group_members = defaultdict(list)
    for i, g in enumerate(groups):
        group_members[g].append(i)
    by_class = defaultdict(list)
    for g, members in group_members.items():
        label = Counter(labels[i] for i in members).most_common(1)[0][0]
        by_class[label].append(g)
    rng = np.random.RandomState(seed)
    fold_of = {}
    offset = 0
    for label in sorted(by_class):
        # order groups by their first document so the split does not depend on
        # how the group ids are numbered
        class_groups = sorted(by_class[label], key=lambda g: group_members[g][0])
        rng.shuffle(class_groups)
        for j, g in enumerate(class_groups):
            fold_of[g] = (j + offset) % k
        offset += len(class_groups)
    return [fold_of[g] for g in groups]


def cross_validate(docs_terms, labels, k=5, alpha=1.0, seed=42, groups=None):
    """k-fold cross-validation; predictions of every fold are pooled."""
    folds = stratified_folds(labels, k, seed, groups)
    y_true, y_pred, fold_accuracy = [], [], []
    for fold in range(k):
        test_idx = [i for i, f in enumerate(folds) if f == fold]
        train_idx = [i for i, f in enumerate(folds) if f != fold]
        if not test_idx or not train_idx:
            continue
        _, t, p, m = train_and_evaluate(docs_terms, labels, train_idx, test_idx, alpha)
        y_true += t
        y_pred += p
        fold_accuracy.append(m["accuracy"])
    metrics = classification_metrics(y_true, y_pred, sorted(set(labels)))
    metrics["fold_accuracy"] = fold_accuracy
    return metrics
