from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt


def plot_mnist_samples(X: np.ndarray, y: np.ndarray, num_samples: int = 5) -> None:
    if X.ndim == 2:
        X = X.reshape(-1, 28, 28)

    classes = np.unique(y)
    plt.figure(figsize=(num_samples * 2, len(classes) * 2))

    for i, cls in enumerate(classes):
        cls_indices = np.where(y == cls)[0]
        selected_indices = np.random.choice(cls_indices, num_samples, replace=False)

        for j, idx in enumerate(selected_indices):
            plt_idx = i * num_samples + j + 1
            plt.subplot(len(classes), num_samples, plt_idx)
            plt.imshow(X[idx], cmap="gray")
            plt.axis("off")
            if j == 0:
                plt.ylabel(cls, fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_class_distributions(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    num_classes: int = 10,
) -> None:
    classes = np.arange(num_classes)
    train_counts = np.bincount(y_train, minlength=num_classes)
    val_counts = np.bincount(y_val, minlength=num_classes)
    test_counts = np.bincount(y_test, minlength=num_classes)

    width = 0.25
    x = np.arange(len(classes))

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, train_counts, width=width, label="Train")
    plt.bar(x, val_counts, width=width, label="Validation")
    plt.bar(x + width, test_counts, width=width, label="Test")
    plt.xlabel("Digit Class")
    plt.ylabel("Number of Samples")
    plt.title("MNIST Class Distributions")
    plt.xticks(classes)
    plt.legend()
    plt.show()


def plot_training_history(history: dict[str, list[float]]) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(history.get("loss", []), label="train loss")
    if history.get("val_loss"):
        plt.plot(history["val_loss"], label="val loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(history.get("acc", []), label="train acc")
    if history.get("val_acc"):
        plt.plot(history["val_acc"], label="val acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | list[int] | None = None,
    title: str = "Confusion Matrix",
) -> None:
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = list(range(num_classes))

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()


def plot_misclassified_examples(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n: int = 12,
    img_shape: tuple[int, int] = (28, 28),
    shuffle: bool = True,
) -> None:
    wrong_idx = np.where(y_true != y_pred)[0]

    if len(wrong_idx) == 0:
        print("No misclassified examples found.")
        return

    if shuffle:
        np.random.shuffle(wrong_idx)

    n = min(n, len(wrong_idx))
    cols = 4
    rows = math.ceil(n / cols)

    plt.figure(figsize=(12, 3 * rows))

    for k, idx in enumerate(wrong_idx[:n]):
        plt.subplot(rows, cols, k + 1)
        img = X[idx]
        if img.ndim == 1:
            img = img.reshape(img_shape)

        plt.imshow(img, cmap="gray")
        plt.title(f"True: {y_true[idx]} | Pred: {y_pred[idx]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
