from __future__ import annotations

import numpy as np


def relative_imbalance(y: np.ndarray, num_classes: int = 10) -> float:
    counts = np.bincount(y, minlength=num_classes)
    total = np.sum(counts)
    diff = np.max(counts) - np.min(counts)
    return float((diff / total) * 100.0)


def to_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2:
        return np.argmax(y, axis=1)
    return y


def build_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 10,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    class_totals = cm.sum(axis=1)
    acc = np.zeros(len(class_totals), dtype=float)

    for i in range(len(class_totals)):
        if class_totals[i] > 0:
            acc[i] = cm[i, i] / class_totals[i]
    return acc


def top_confusion_pairs(
    cm: np.ndarray,
    top_k: int = 5,
    symmetric: bool = True,
) -> list[tuple[int, int, int]]:
    pairs: list[tuple[int, int, int]] = []

    if symmetric:
        for i in range(cm.shape[0]):
            for j in range(i + 1, cm.shape[1]):
                count = int(cm[i, j] + cm[j, i])
                pairs.append((i, j, count))
    else:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j:
                    pairs.append((i, j, int(cm[i, j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]
